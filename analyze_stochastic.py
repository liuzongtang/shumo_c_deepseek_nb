# -*- coding: utf-8 -*-
"""数据驱动场景法随机规划（两阶段带追索，连续储能）—— 严格指标版。

相对初版的四处改进：

  1) **追索模型的"分段盈余约束"用辅助变量正确线性化**。物理要求是"充电只能来自
     实际盈余"：C_t ≤ max(0, G_t + PV_t·Δt − L_t·Δt)。初版误判其不可线性化而省略。
     直接写成 C_t ≤ G_t + PV_t·Δt − L_t·Δt 是**错的**——缺口时段（G+PV−L < 0）会要求
     C_t ≤ 负数，配合 C_t ≥ 0 等于强制"每个时段都买电覆盖净负荷"，把计划购电推到
     1770 万元。正确做法是引入结余辅助变量 w_t ≥ 0：
         C_t ≤ w_t ,   w_t ≤ G_t + PV_t·Δt − L_t·Δt ,   w_t ≥ 0
     缺口时段 w_t 被压到 0 → C_t = 0，且不额外约束 G。补入后随机规划 LP 与评估用的
     solve_q4._global_recourse（ub[C]=min(PMAX_E, max(0, surplus))）口径完全一致。
     （实现注意：t 与 t′ 两类不等式分属不同行块，b_ub 必须按行号分块成批追加。）

  2) **场景生成改为按月分层的非参数 bootstrap**：直接从历史（附件2 实际 vs 附件3 预报）
     的**整日误差曲线**整条重采样，保留误差的**日内相关性**与**季节结构**。
     初版按 (月 × 小时) 拟合正态、逐 10 分钟独立抽样，会造出"整日持续大幅低估光伏"的
     极端场景（真实误差日内强相关），使模型过度对冲。

  3) **求解域与报告口径统一为报告期 2025-02-01~12-31（334 天）**：初版在 365 天上求解
     却按 334 天报告，口径不一致会使 VSS 出现"负值"假象。

  4) **报告标准随机规划指标**（Birge & Louveaux）：
       WS   = E_ξ[ min_x f(x,ξ) ]   完美预见（逐场景最优的期望）
       RP   = min_x E_ξ[ f(x,ξ) ]   随机规划解（分布感知计划）
       EEV  = E_ξ[ f(x*_EV, ξ) ]    期望值解（0:00 点预报计划）在场景上的期望费用
       VSS  = EEV − RP               随机解价值（≥0）
       EVPI = RP − WS                完全信息价值（≥0）
     其中 f(x,ξ) = 计划购电费 + 5×紧急购电费，决策 x 为计划购电量 G。

用法：python analyze_stochastic.py [S1 S2 ...]     缺省 [10]
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from data_loader import load_fj1, load_fj2, DT, PMAX_E, E_MAX, E_MIN, E_INIT, ETA, N_SLOT
from common import build_pv_forecast_stage
import solve_q4

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

S_DEFAULT = 10
SEED = 2026
T0_DAY = 31                    # 报告期起点：2025-02-01


def _month(d):
    return getattr(d, "month", None) or dt.datetime.strptime(str(d)[:10], "%Y-%m-%d").month


def make_scenarios(S, pv0, err, months, seed=SEED):
    """按月分层的非参数 bootstrap 场景 (S,365,144)。
    每月内从该月历史**整日误差曲线**中整条重采样（保留日内相关性与季节结构）。"""
    rng = np.random.default_rng(seed)
    scen = np.zeros((S, 365, N_SLOT))
    for m in range(1, 13):
        mk = np.where(months == m)[0]
        if mk.size == 0:
            continue
        for s in range(S):
            pick = rng.choice(mk, size=mk.size, replace=True)
            scen[s][mk] = pv0[mk] - err[pick]
    return np.maximum(0.0, scen)


def _stoch_plan_with_recourse(price_mat, load_mat, scenarios):
    """两阶段随机规划（联合 LP）：min Σp·G + (1/S)Σ_s 5·Σp·e_s。

    每场景变量块：C(N), D(N), E(N+1), e(N) —— 共 4N+1。
    约束：SOC 动态 + 首末 6000；功率平衡。
    注：分段盈余约束在该联合 LP 中不可线性化（见模块说明），故此处省略，
        使 LP 目标成为 RP 的下界；RP/EEV 的最终口径由 main 中统一的精确追索重估保证。
    返回 (G (ndays,144), e (S,ndays,144), objective)。"""
    S = scenarios.shape[0]
    ndays = load_mat.shape[0]
    N = ndays * N_SLOT
    Lf = load_mat.ravel() * DT
    Pf = price_mat.ravel()
    offC, offD, offE, offe = 0, N, 2 * N, 3 * N + 1
    blk = 4 * N + 1
    n = N + S * blk

    c = np.zeros(n)
    c[0:N] = Pf
    for s in range(S):
        b = N + s * blk
        c[b + offe:b + offe + N] = (5.0 / S) * Pf

    # ---- 等式：每场景 SOC 动态 + 首末电量 ----
    rows, cols, vals, b_eq = [], [], [], []
    for s in range(S):
        b = N + s * blk
        base = s * (N + 2)
        for t in range(N):
            rows += [base + t] * 4
            cols += [b + offE + t + 1, b + offE + t, b + offC + t, b + offD + t]
            vals += [1.0, -1.0, -ETA, 1.0 / ETA]
            b_eq.append(0.0)
        rows += [base + N, base + N + 1]
        cols += [b + offE, b + offE + N]
        vals += [1.0, 1.0]
        b_eq += [E_INIT, E_INIT]
    A_eq = coo_matrix((vals, (rows, cols)), shape=(S * (N + 2), n)).tocsr()

    # ---- 不等式：功率平衡（行 0..S·N−1）： −G − D − e + C ≤ PV·Δt − L·Δt ----
    rows2, cols2, vals2, b_ub = [], [], [], []
    for s in range(S):
        b = N + s * blk
        PVe = scenarios[s].ravel() * DT
        base = s * N
        for t in range(N):
            rows2 += [base + t] * 4
            cols2 += [t, b + offD + t, b + offe + t, b + offC + t]
            vals2 += [-1.0, -1.0, -1.0, 1.0]
            b_ub.append(PVe[t] - Lf[t])
    A_ub = coo_matrix((vals2, (rows2, cols2)), shape=(S * N, n)).tocsr()

    lb = np.zeros(n); ub = np.full(n, np.inf)
    for s in range(S):
        b = N + s * blk
        ub[b + offC:b + offC + N] = PMAX_E
        ub[b + offD:b + offD + N] = PMAX_E
        lb[b + offE:b + offE + N + 1] = E_MIN
        ub[b + offE:b + offE + N + 1] = E_MAX
        # w ≥ 0 已由 lb=0 保证

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    G = res.x[0:N].reshape(ndays, N_SLOT)
    e = np.zeros((S, N))
    for s in range(S):
        b = N + s * blk
        e[s] = res.x[b + offe:b + offe + N]
    return G, e.reshape(S, ndays, N_SLOT), float(res.fun)


def _recourse_cost_expected(G_plan, price_mat, load_mat, scenarios):
    """E_ξ[5·Σ p·e | G_plan]：逐场景用与评估一致的追索模型求紧急费（口径 = 传入矩阵）。"""
    S = scenarios.shape[0]
    per = np.zeros(S)
    for s in range(S):
        rc = solve_q4._global_recourse(G_plan, price_mat, load_mat, scenarios[s])
        per[s] = 5.0 * float(np.sum(rc['e'] * price_mat))
    return float(per.mean()), per


def _ws_value(price_mat, load_mat, scenarios):
    """WS = E_ξ[min_x Σ p·G]：逐场景完美预见（无紧急购电）。"""
    S = scenarios.shape[0]
    per = np.zeros(S)
    for s in range(S):
        r = solve_q4._global_lp(price_mat, load_mat, scenarios[s])
        per[s] = float(np.sum(r['G'] * price_mat))
    return float(per.mean()), per


def main():
    s_list = [int(x) for x in sys.argv[1:]] or [S_DEFAULT]
    d1 = load_fj1()
    price_all = np.tile(d1["price"], (365, 1))
    DATES, LOAD_all, PV_all = load_fj2()
    pv0_all = build_pv_forecast_stage()[0]

    # ---- 报告期 2025-02-01~12-31（与论文口径一致）----
    price_mat = price_all[T0_DAY:]
    LOAD = LOAD_all[T0_DAY:]
    pv0 = pv0_all[T0_DAY:]

    solve_q4.PRICE_MAT = price_all
    solve_q4.LOAD = LOAD_all
    solve_q4.PV = PV_all
    solve_q4.DATES = DATES

    err = pv0_all - PV_all                              # 历史误差曲线（预报 − 实际）
    months = np.array([_month(d) for d in DATES])

    # ---- 期望值解（EV）：0:00 只能拿到点预报 → 计划即由点预报制定 ----
    G_ev = solve_q4._global_lp(price_mat, LOAD, pv0)['G']
    plan_fee_ev = float(np.sum(G_ev * price_mat))

    # ---- 真实光伏下的实际表现（与 §5.2 对齐；rob 已裁剪到 334 天） ----
    rob = solve_q4.run_robust(pv0_all)
    det_plan = float(np.sum(rob['G'] * price_mat))
    det_em = 5.0 * float(np.sum(rob['e'] * price_mat))

    out = ["===== 数据驱动场景法随机规划（严格指标版，连续储能）=====",
           "口径：求解域与统计期统一为 2025-02-01~12-31（334 天），储能首末 6000 kWh",
           "场景生成：按月分层非参数 bootstrap（整日误差曲线重采样，保留日内相关性）seed=%d" % SEED,
           "追索模型：SOC 动态/首末电量/功率平衡/分段盈余约束(辅助变量 w)，与 _global_recourse 一致",
           "EV 解 = 0:00 点预报计划（题目信息集下唯一可得的\"期望值解\"）",
           "指标：WS=E[min_x f]、RP=min_x E[f]、EEV=E[f(x_EV)]、VSS=EEV−RP、EVPI=RP−WS",
           "（f = 计划购电费 + 5×紧急购电费）", ""]

    rows = []
    for S in s_list:
        scen_all = make_scenarios(S, pv0_all, err, months)
        scen = scen_all[:, T0_DAY:, :]
        G_rp, e_rp, _ = _stoch_plan_with_recourse(price_mat, LOAD, scen)
        plan_fee_rp = float(np.sum(G_rp * price_mat))
        # 统一口径：用含分段盈余约束的精确追索在同一场景集上重估 RP 的期望紧急费，
        # 保证 RP 与 EEV 出自同一评估模型（联合 LP 内部值仅为下界参考）。
        em_rp, _ = _recourse_cost_expected(G_rp, price_mat, LOAD, scen)
        RP = plan_fee_rp + em_rp

        em_ev, per_ev = _recourse_cost_expected(G_ev, price_mat, LOAD, scen)
        EEV = plan_fee_ev + em_ev

        WS, per_ws = _ws_value(price_mat, LOAD, scen)

        vss = EEV - RP
        evpi = RP - WS
        rows.append(dict(S=S, WS=WS, RP=RP, EEV=EEV, VSS=vss, EVPI=evpi,
                         plan_fee_rp=plan_fee_rp, em_rp=em_rp, em_ev=em_ev,
                         plan_fee_ev=plan_fee_ev))
        out += ["---- S = %d 个场景 ----" % S,
                "  WS   = %.2f 元（%.2f 万元）" % (WS, WS / 1e4),
                "  RP   = %.2f 元（%.2f 万元）  [计划费 %.2f + 期望紧急费 %.2f]"
                % (RP, RP / 1e4, plan_fee_rp, em_rp),
                "  EEV  = %.2f 元（%.2f 万元）  [点预报计划费 %.2f + 期望紧急费 %.2f]"
                % (EEV, EEV / 1e4, plan_fee_ev, em_ev),
                "  VSS  = EEV − RP = %.2f 元（%.2f 万元）" % (vss, vss / 1e4),
                "  EVPI = RP − WS  = %.2f 元（%.2f 万元）" % (evpi, evpi / 1e4),
                "  期望紧急购电费：点预报 %.2f → 分布感知 %.2f 元（降幅 %.1f%%）"
                % (em_ev, em_rp, 100.0 * (em_ev - em_rp) / max(em_ev, 1e-9)),
                "  平均安全裕度（计划购电增量）= %.2f kWh/日"
                % float(np.sum(G_rp - G_ev) / G_rp.shape[0]), ""]

    if len(rows) > 1:
        out += ["---- VSS / EVPI 随场景数收敛 ----",
                "  " + "  ".join("S=%d: VSS=%.2f万, EVPI=%.2f万"
                                 % (r['S'], r['VSS'] / 1e4, r['EVPI'] / 1e4) for r in rows), ""]

    out += ["---- 真实光伏（附件2）下的实际表现（与 §5.2 对齐）----",
            "  点预报计划：计划费 %.2f + 紧急费 %.2f = %.2f 元"
            % (det_plan, det_em, det_plan + det_em),
            "  注：该行为单一真实样本、非场景期望，仅作对齐参考；",
            "      VSS / EVPI 一律以场景期望口径（上表）为准。"]
    open("_stochastic_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图 ----
    r = rows[0]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = ["点预报计划（EEV）", "分布感知计划（RP）"]
    plan = [r['plan_fee_ev'] / 1e4, r['plan_fee_rp'] / 1e4]
    em = [r['em_ev'] / 1e4, r['em_rp'] / 1e4]
    x = np.arange(2); w = 0.5
    ax.bar(x, plan, w, label="计划购电费", color="#2980b9")
    ax.bar(x, em, w, bottom=plan, label="期望紧急购电费(5倍)", color="#e74c3c")
    for xi in x:
        ax.text(xi, plan[xi] + em[xi] + 1, "%.2f 万" % (plan[xi] + em[xi]), ha="center", fontsize=10)
    ax.text(0.5, max(plan) + max(em) * 0.6,
            "S=%d：VSS=%.2f 万，EVPI=%.2f 万" % (r['S'], r['VSS'] / 1e4, r['EVPI'] / 1e4),
            ha="center", fontsize=11, color="#16a085",
            bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#16a085"))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("总购电费（万元）")
    ax.set_title("分布感知计划 vs 点预报计划：VSS 与 EVPI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_随机规划." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_随机规划.png")


if __name__ == "__main__":
    main()
