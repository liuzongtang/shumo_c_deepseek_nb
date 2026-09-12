# -*- coding: utf-8 -*-
"""问题三深化：预报时刻边际价值（状态反馈/闭环 MPC 口径）。

定量回答题目"是否需引入其他时刻预报"：在**状态反馈滚动**框架下逐次加入 6/12/18 时
预报，度量每个时刻的边际降费价值，并用"完美日内预报"给出额外时刻价值的上界。

与问题三主模型一致：6/12/18 再优化的初始储电量由"已下发计划 + 实际光伏"因果递推
（common.advance_actual_soc），构成严格的状态反馈 MPC。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, load_fj2, N_SLOT, DT
from common import build_pv_forecast_stage, stage_lp, advance_actual_soc
import solve_q4

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def rolling_state_feedback(pvf, times, pv_override=None):
    """状态反馈滚动。times: 升序调整小时列表(不含0，如[6,12,18])。
    SOC 交接用 advance_actual_soc（实际光伏推进），跨日边界仍用全局预报 SOC。
    pv_override: dict {hour:(365,144)}，提供时该时刻的重优化用该 PV 作"完美日内预报"。
    返回 G_plan, G_adj (365,144)。"""
    E_plan = solve_q4._global_lp(solve_q4.PRICE_MAT, solve_q4.LOAD, pvf[0])['E']
    ndays = 365
    G_plan = np.zeros((ndays, N_SLOT))
    G_adj = np.zeros((ndays, N_SLOT))
    for d in range(ndays):
        p = solve_q4.PRICE_MAT[d]
        ld = solve_q4.LOAD[d]
        pv_act = solve_q4.PV[d]
        E0b = E_plan[d * N_SLOT]
        E1b = E_plan[(d + 1) * N_SLOT]
        r0 = stage_lp(p, ld, pvf[0][d], 0, E0b, E1b, None)
        G_plan[d] = r0['G']
        # 已承诺的调度（全 144 段长度），随每个时点更新
        G_commit = r0['G'].copy()
        C_commit = r0['C'].copy()
        D_commit = r0['D'].copy()
        E_cur = E0b
        prev_t = 0
        for tau in times:
            t0 = tau * 6
            seg = slice(prev_t * 6, t0)
            E_hand, _ = advance_actual_soc(pv_act[seg], ld[seg], G_commit[seg],
                                           C_commit[seg], D_commit[seg], E_cur)
            fv = pvf[tau][d] if pv_override is None else pv_override[tau][d]
            r = stage_lp(p, ld, fv, t0, E_hand, E1b, G_plan[d][t0:])
            G_commit[t0:] = r['G']
            C_commit[t0:] = r['C']
            D_commit[t0:] = r['D']
            E_cur = E_hand
            prev_t = tau
        G_adj[d] = G_commit
    return G_plan, G_adj


def _cost(G_plan, G_adj, p):
    """滚动总费 = 计划费 + 调整费(1.5·up+0.5·dn) + 5×紧急费（334 天报告期）。"""
    rc = solve_q4._global_recourse(G_adj, solve_q4.PRICE_MAT, solve_q4.LOAD, solve_q4.PV)
    sl = slice(31, 365)
    plan = float(np.sum(G_plan[sl] * p))
    up = np.maximum(0.0, G_adj - G_plan)
    dn = np.maximum(0.0, G_plan - G_adj)
    adj = float(np.sum((1.5 * up[sl] + 0.5 * dn[sl]) * p))
    em = 5.0 * float(np.sum(rc['e'][sl] * p))
    return plan, adj, em, plan + adj + em


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
    out.append("===== 问题三深化：预报时刻边际价值（状态反馈 MPC 口径）=====")
    out.append("")

    # ---- 预报时刻边际价值 ----
    configs = [([], "0:00 仅计划（=问题2鲁棒）"),
               ([6], "0:00+6:00"),
               ([6, 12], "0:00+6:00+12:00"),
               ([6, 12, 18], "0:00+6:00+12:00+18:00（=问题3）")]
    results = []
    for times, name in configs:
        G_plan, G_adj = rolling_state_feedback(pvf, times)
        plan, adj, em, total = _cost(G_plan, G_adj, p)
        results.append((name, plan, adj, em, total))
        out.append("  %-28s 计划费 %.2f + 调整费 %.2f + 紧急费 %.2f = %.2f 元"
                   % (name, plan, adj, em, total))

    # 完美日内预报（上界：6/12/18 的重优化用实际光伏）
    G_plan, G_adj_perf = rolling_state_feedback(pvf, [6, 12, 18],
                                                pv_override={6: pv, 12: pv, 18: pv})
    plan_p, adj_p, em_p, total_p = _cost(G_plan, G_adj_perf, p)
    out.append("  %-28s 计划费 %.2f + 调整费 %.2f + 紧急费 %.2f = %.2f 元"
               % ("完美日内预报(6/12/18=实际)", plan_p, adj_p, em_p, total_p))

    out.append("")
    out.append("  逐次加入预报时刻的边际降费价值：")
    prev = results[0][4]
    for name, _, _, _, total in results[1:]:
        out.append("    %-26s 再降 %.2f 元（%.2f 万元）" % (name, prev - total, (prev - total) / 1e4))
        prev = total
    gap_perfect = results[-1][4] - total_p
    out.append("  额外时刻/精度价值上界（问题3 → 完美日内预报）= %.2f 元（%.2f 万元）"
               % (gap_perfect, gap_perfect / 1e4))
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
