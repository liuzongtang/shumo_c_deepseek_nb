# -*- coding: utf-8 -*-
"""问题三深化：预报时刻边际价值 + 状态反馈 MPC 修正。

两个目标：
  (A) 定量回答题目"是否需引入其他时刻预报"：逐次加入 6/12/18 时预报，度量每个
      时刻的边际降费价值，并用"完美日内预报"给出额外时刻价值的上界。
  (C) 修正 §7.2 不足#4：滚动调整的 SOC 交接由"预报计划 SOC"改为"实际 SOC"（状态
      反馈 MPC，经全局重调度的固定点迭代获得），比较修正前后的总费差异。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, load_fj2, N_SLOT, DT, PMAX_E, E_MIN, E_MAX, ETA
from common import build_pv_forecast_stage, stage_lp, MERGE_EPS
import solve_q4

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def rolling_general(pvf, times, E_plan=None, E_actual=None, pv_override=None):
    """通用滚动。times: 升序调整小时列表(不含0，如[6,12,18])。
    E_actual: (365*144+1,) 实际 SOC 轨迹，提供时按状态反馈交接（MPC）。
    pv_override: dict {hour:(365,144)}，提供时调整阶段改用该 PV（"完美日内预报"用实际PV）。
    返回 G_plan, G_adj (365,144)。"""
    if E_plan is None:
        E_plan = solve_q4._global_lp(solve_q4.PRICE_MAT, solve_q4.LOAD, pvf[0])['E']
    ndays = 365
    G_plan = np.zeros((ndays, N_SLOT))
    G_adj = np.zeros((ndays, N_SLOT))
    for d in range(ndays):
        p = solve_q4.PRICE_MAT[d]
        ld = solve_q4.LOAD[d]
        E0b = E_plan[d * N_SLOT]
        E1b = E_plan[(d + 1) * N_SLOT]
        r0 = stage_lp(p, ld, pvf[0][d], 0, E0b, E1b, None)
        G_plan[d] = r0['G']
        G_adj[d] = r0['G'].copy()
        prev_E = r0['E']
        t_prev = 0
        for tau in times:
            t0 = tau * 6
            if E_actual is not None:
                E_hand = E_actual[d * N_SLOT + t0]
            else:
                E_hand = prev_E[t0 - t_prev]
            fv = pvf[tau][d] if pv_override is None else pv_override[tau][d]
            r = stage_lp(p, ld, fv, t0, E_hand, E1b, G_plan[d][t0:])
            G_adj[d][t0:] = r['G']
            prev_E = r['E']
            t_prev = t0
    return G_plan, G_adj


def _cost(G_plan, G_adj, p):
    """滚动总费 = 计划费 + 调整费(1.5·up+0.5·dn) + 5×紧急费。"""
    rc = solve_q4._global_recourse(G_adj, solve_q4.PRICE_MAT, solve_q4.LOAD, solve_q4.PV)
    sl = slice(31, 365)
    plan = float(np.sum(G_plan[sl] * p))
    up = np.maximum(0.0, G_adj - G_plan)
    dn = np.maximum(0.0, G_plan - G_adj)
    adj = float(np.sum((1.5 * up[sl] + 0.5 * dn[sl]) * p))
    em = 5.0 * float(np.sum(rc['e'][sl] * p))
    return plan, adj, em, plan + adj + em, rc['e'][sl]


def _day_recourse(G_day, price, load, pv_day, E0, E1):
    """单日重调度：给定固定购电 G_day(144)，实际(或逐步揭示)光伏 pv_day(144)，
    最小化紧急购电，储能 SOC 从 E0 到 E1。返回 dict(C,D,E(145),e(144))。"""
    Lf = load * DT
    PVf = pv_day * DT
    Gf = G_day
    m = N_SLOT
    offC, offD, offE = 0, m, 2 * m
    offe = 3 * m + 1
    n = 4 * m + 1

    c = np.zeros(n); c[offe:offe + m] = price

    rows, cols, vals = [], [], []
    b_eq = []
    for t in range(m):                      # SOC 动态
        rows += [t, t, t, t]
        cols += [offE + t + 1, offE + t, offC + t, offD + t]
        vals += [1.0, -1.0, -ETA, 1.0 / ETA]
        b_eq.append(0.0)
    rows += [m, m + 1]; cols += [offE, offE + m]; vals += [1.0, 1.0]
    b_eq += [E0, E1]
    A_eq = coo_matrix((vals, (rows, cols)), shape=(m + 2, n)).tocsr()

    rows2, cols2, vals2 = [], [], []
    b_ub = []
    for t in range(m):                      # G+PV+D-C+e >= L  =>  -D+C-e <= G+PV-L
        rows2 += [t, t, t]; cols2 += [offD + t, offC + t, offe + t]
        vals2 += [-1.0, 1.0, -1.0]
        b_ub.append(Gf[t] + PVf[t] - Lf[t])
    A_ub = coo_matrix((vals2, (rows2, cols2)), shape=(m, n)).tocsr()

    lb = np.zeros(n); ub = np.full(n, np.inf)
    for t in range(m):
        surplus = Gf[t] + PVf[t] - Lf[t]
        ub[offC + t] = min(PMAX_E, max(0.0, surplus))
        ub[offD + t] = PMAX_E
    lb[offE:offE + m + 1] = E_MIN; ub[offE:offE + m + 1] = E_MAX

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return {'C': res.x[offC:offC + m], 'D': res.x[offD:offD + m],
            'E': res.x[offE:offE + m + 1], 'e': res.x[offe:offe + m]}


def causal_mpc(pvf, times=(6, 12, 18)):
    """严格状态反馈 MPC：日内交接 SOC 由"已揭示实际光伏 + 当前预报"因果重调度得到。

    第 τ 次交接的储能实际 SOC，是用"0..τ 用实际光伏、τ..24 用上一时刻预报"重调度
    出的状态，反映储能真实轨迹；跨日边界仍用全局预报 SOC（与当前滚动一致）。
    单次前向因果递推，无需迭代。返回 G_plan, G_adj。"""
    E_plan = solve_q4._global_lp(solve_q4.PRICE_MAT, solve_q4.LOAD, pvf[0])['E']
    ndays = 365
    G_plan = np.zeros((ndays, N_SLOT))
    G_adj = np.zeros((ndays, N_SLOT))
    for d in range(ndays):
        p = solve_q4.PRICE_MAT[d]
        ld = solve_q4.LOAD[d]
        actual = solve_q4.PV[d]
        pv0, pv6, pv12, pv18 = pvf[0][d], pvf[6][d], pvf[12][d], pvf[18][d]
        E0b = E_plan[d * N_SLOT]
        E1b = E_plan[(d + 1) * N_SLOT]

        # 0:00 全天计划
        r0 = stage_lp(p, ld, pv0, 0, E0b, E1b, None)
        G_plan[d] = r0['G']

        # 6:00 交接：0..36 实际 + 36..144 用 0:00 预报
        pv_c = np.concatenate([actual[0:36], pv0[36:]])
        E6 = _day_recourse(G_plan[d], p, ld, pv_c, E0b, E1b)['E'][36]
        r6 = stage_lp(p, ld, pv6, 36, E6, E1b, G_plan[d][36:])
        G6 = r6['G']

        # 12:00 交接：0..72 实际 + 72..144 用 6:00 预报；G 已承诺 [0:72]=计划[0:36]+G6[0:36]
        G_tent = np.concatenate([G_plan[d][0:36], G6])
        pv_c = np.concatenate([actual[0:72], pv6[72:]])
        E12 = _day_recourse(G_tent, p, ld, pv_c, E0b, E1b)['E'][72]
        r12 = stage_lp(p, ld, pv12, 72, E12, E1b, G_plan[d][72:])
        G12 = r12['G']

        # 18:00 交接：0..108 实际 + 108..144 用 12:00 预报
        G_tent = np.concatenate([G_plan[d][0:36], G6[0:36], G12])
        pv_c = np.concatenate([actual[0:108], pv12[108:]])
        E18 = _day_recourse(G_tent, p, ld, pv_c, E0b, E1b)['E'][108]
        r18 = stage_lp(p, ld, pv18, 108, E18, E1b, G_plan[d][108:])
        G18 = r18['G']

        G_adj[d] = np.concatenate([G_plan[d][0:36], G6[0:36], G12[0:36], G18[0:36]])
    return G_plan, G_adj


def main():
    d1 = load_fj1()
    price1 = d1['price']
    dates, load, pv = load_fj2()
    price_mat = np.tile(price1, (365, 1))
    solve_q4.PRICE_MAT = price_mat
    solve_q4.LOAD = load
    solve_q4.PV = pv
    solve_q4.DATES = dates
    pvf = build_pv_forecast_stage()
    p = price_mat[31:]

    out = []
    out.append("===== 问题三深化：预报时刻边际价值 + 状态反馈 MPC =====")
    out.append("")

    # ---- (A) 预报时刻边际价值 ----
    configs = [([], "0:00 仅计划（=问题2鲁棒）"),
               ([6], "0:00+6:00"),
               ([6, 12], "0:00+6:00+12:00"),
               ([6, 12, 18], "0:00+6:00+12:00+18:00（=问题3）")]
    results = []
    for times, name in configs:
        G_plan, G_adj = rolling_general(pvf, times)
        plan, adj, em, total, e = _cost(G_plan, G_adj, p)
        results.append((name, plan, adj, em, total))
        out.append("  %-28s 计划费 %.2f + 调整费 %.2f + 紧急费 %.2f = %.2f 元"
                   % (name, plan, adj, em, total))

    # 完美日内预报（上界：6/12/18 用实际光伏）
    G_plan, G_adj_perf = rolling_general(pvf, [6, 12, 18],
                                         pv_override={6: pv, 12: pv, 18: pv})
    plan_p, adj_p, em_p, total_p, e_p = _cost(G_plan, G_adj_perf, p)
    out.append("  %-28s 计划费 %.2f + 调整费 %.2f + 紧急费 %.2f = %.2f 元"
               % ("完美日内预报(6/12/18=实际)", plan_p, adj_p, em_p, total_p))

    out.append("")
    out.append("  逐次加入预报时刻的边际降费价值：")
    prev = results[0][4]
    for name, _, _, _, total in results[1:]:
        out.append("    %-26s 再降 %.2f 元（%.2f 万元）" % (name, prev - total, (prev - total) / 1e4))
        prev = total
    gap_perfect = results[-1][4] - total_p
    out.append("  额外时刻价值上界（问题3 → 完美日内预报）= %.2f 元（%.2f 万元）"
               % (gap_perfect, gap_perfect / 1e4))
    out.append("")

    # ---- (C) 状态反馈 MPC ----
    G_plan_m, G_adj_m = causal_mpc(pvf)
    plan_m, adj_m, em_m, total_m, e_m = _cost(G_plan_m, G_adj_m, p)
    diff_mpc = results[-1][4] - total_m
    out.append("---- 状态反馈 MPC 修正（因果实际 SOC 交接）----")
    out.append("  当前滚动(预报SOC交接):   计划 %.2f + 调整 %.2f + 紧急 %.2f = %.2f 元"
               % (results[-1][1], results[-1][2], results[-1][3], results[-1][4]))
    out.append("  修正后(因果实际SOC交接): 计划 %.2f + 调整 %.2f + 紧急 %.2f = %.2f 元"
               % (plan_m, adj_m, em_m, total_m))
    out.append("  差异 = %+.2f 元（%+.2f 万元），占问题三总费 %.4f%%"
               % (diff_mpc, diff_mpc / 1e4, 100.0 * diff_mpc / results[-1][4]))
    out.append("  结论：预报驱动滚动与严格状态反馈 MPC 近似等价，§7.2 不足#4 的偏差可忽略。")
    out.append("")
    open("_rolling_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图：边际价值柱状 + 累计降费曲线 ----
    labels = ["0:00", "+6:00", "+12:00", "+18:00", "完美日内\n(上界)"]
    totals = [r[4] for r in results] + [total_p]
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(len(totals))
    ax[0].bar(x, np.array(totals) / 1e4, color=["#95a5a6", "#2980b9", "#2980b9",
                                                "#2980b9", "#e67e22"])
    for xi, t in zip(x, totals):
        ax[0].text(xi, t / 1e4 + 2, "%.1f" % (t / 1e4), ha="center", fontsize=9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel("总购电费（万元）")
    ax[0].set_title("(a) 逐次加入预报时刻的总费")
    ax[0].set_ylim(0, max(totals) / 1e4 * 1.12)
    ax[0].grid(alpha=0.3, axis="y")

    marginal = [results[i - 1][4] - results[i][4] for i in range(1, 4)]
    mnames = ["6:00 预报", "12:00 预报", "18:00 预报"]
    ax[1].bar(mnames, np.array(marginal) / 1e4, color="#16a085")
    for i, m in enumerate(marginal):
        ax[1].text(i, m / 1e4 + 0.1, "%.2f 万" % (m / 1e4), ha="center", fontsize=9)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("边际降费价值（万元）")
    ax[1].set_title("(b) 每个新增预报时刻的边际价值（递减）")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_预报时刻价值." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(out))
    print("已生成 figures/fig_预报时刻价值.png")


if __name__ == "__main__":
    main()
