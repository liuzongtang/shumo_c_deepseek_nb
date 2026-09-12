# -*- coding: utf-8 -*-
"""P2 扩展：负载不确定性下的分布感知计划（**二维**不确定：负载 + 光伏）。

题目只给出光伏预报（附件3），未提供负载预报。工程上短期负载预报误差通常在 $2\%\sim5\%$，
本文据此假设："0:00 拿到的负载预报 = 实际负载 × (1+ε)"，其中 ε 为**按日相关**的负载
预报误差（每日一个 ε ~ N(0, σ_L)），把它与光伏的按月分层 bootstrap 场景组合，构成
**二维不确定集**，在 §6.5 的两阶段随机规划框架下求解，量化"负载不确定性"的额外代价。

对照：σ_L = 0（仅光伏不确定）应复现 §6.5 的 RP（1305.37 万元）；σ_L = 3% 为工程典型值。

  σ_L = 0 时，本脚本的 RP 即为 §6.5 的 RP（同口径、同随机种子），可交叉验证。

口径：全年 365 天连续储能、首末 6000 kWh；统计 2025-02-01~12-31（334 天）；
分段盈余约束在联合 LP 中不可线性化故省略 → RP 为下界；指标由精确追索模型重估。

用法：python analyze_load_uncertainty.py [sigma_L ...]     缺省 0 0.03
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from data_loader import load_fj1, load_fj2, DT, PMAX_E, E_MAX, E_MIN, E_INIT, ETA, N_SLOT
from common import build_pv_forecast_stage
import solve_q4
import analyze_stochastic as A

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
T0 = A.T0_DAY
SEED_L = 777


def _stoch_plan_2d(price_mat, load_scens, pv_scens):
    """二维不确定的两阶段随机规划（联合 LP）：
        min Σp·G + (1/S)Σ_s 5·Σp·e_s
    每场景拥有**各自的负载与光伏**；变量 G(N) + 每场景 C(N),D(N),E(N+1),e(N)。
    只有一个不等式块（功率平衡），b_ub 与行号天然一致。
    返回 (G (ndays,144), objective)。"""
    S = pv_scens.shape[0]
    ndays = pv_scens.shape[1]
    N = ndays * N_SLOT
    Pf = price_mat.ravel()
    offC, offD, offE, offe = 0, N, 2 * N, 3 * N + 1
    blk = 4 * N + 1
    n = N + S * blk

    c = np.zeros(n)
    c[0:N] = Pf
    for s in range(S):
        b = N + s * blk
        c[b + offe:b + offe + N] = (5.0 / S) * Pf

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

    rows2, cols2, vals2, b_ub = [], [], [], []
    for s in range(S):
        b = N + s * blk
        Lf = load_scens[s].ravel() * DT                 # 该场景自己的负载
        PVe = pv_scens[s].ravel() * DT                  # 该场景自己的光伏
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

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[0:N].reshape(ndays, N_SLOT), float(res.fun)


def _expected_emergency_2d(G, price_mat, load_scens, pv_scens):
    """E_ξ[5Σ_{t≥31} p·e | G]：逐场景用**其自身的负载与光伏**做精确追索。"""
    S = pv_scens.shape[0]
    p = price_mat[T0:]
    per = np.zeros(S)
    for s in range(S):
        rc = solve_q4._global_recourse(G[T0:], p, load_scens[s][T0:], pv_scens[s][T0:])
        per[s] = 5.0 * float(np.sum(rc['e'] * p))
    return float(per.mean()), per


def main():
    sig_list = [float(x) for x in sys.argv[1:]] or [0.0, 0.03]
    S = 10
    price_all = np.tile(load_fj1()["price"], (365, 1))
    DATES, LOAD_all, PV_all = load_fj2()
    pv0_all = build_pv_forecast_stage()[0]
    err = pv0_all - PV_all
    months = np.array([A._month(d) for d in DATES])
    scen_pv_all = A.make_scenarios(S, pv0_all, err, months)     # 光伏场景（按月分层 bootstrap）
    p31 = price_all[T0:]

    # EV 解（点预报计划）：负载与光伏都取点预报
    G_ev = solve_q4._global_lp(price_all, LOAD_all, pv0_all)['G']
    plan_fee_ev = float(np.sum(G_ev[T0:] * p31))

    out = ["===== 负载不确定性下的分布感知计划（二维不确定，S=%d）=====" % S,
           "负载不确定假设：0:00 负载预报 = 实际负载 ×(1+ε)，ε 按日相关 ~ N(0, σ_L)",
           "光伏不确定：沿用 §6.5 的按月分层非参数 bootstrap 场景",
           "口径：365 天连续储能/首末 6000 kWh，统计 334 天；指标由精确追索重估",
           "（分段盈余约束在联合 LP 中不可线性化，故 RP 为下界、VSS 为上界）", ""]

    rows = []
    for sig in sig_list:
        rng = np.random.default_rng(SEED_L)
        eps = rng.normal(0.0, sig, size=(S, 365)) if sig > 0 else np.zeros((S, 365))
        load_scens = np.maximum(LOAD_all[None, :, :] * (1.0 + eps[:, :, None]), 0.0)
        G, obj = _stoch_plan_2d(price_all, load_scens, scen_pv_all)
        plan_fee = float(np.sum(G[T0:] * p31))
        em, per = _expected_emergency_2d(G, price_all, load_scens, scen_pv_all)
        RP = plan_fee + em

        em_ev, per_ev = _expected_emergency_2d(G_ev, price_all, load_scens, scen_pv_all)
        EEV = plan_fee_ev + em_ev
        vss = EEV - RP
        rows.append(dict(sig=sig, plan_fee=plan_fee, em=em, RP=RP, EEV=EEV, VSS=vss,
                         em_ev=em_ev, plan_fee_ev=plan_fee_ev,
                         margin=float(np.sum(G[T0:] - G_ev[T0:]) / G[T0:].shape[0])))
        out += ["---- σ_L = %.1f%% ----" % (sig * 100),
                "  RP   = %.2f 元（%.2f 万元）  [计划费 %.2f + 期望紧急费 %.2f]"
                % (RP, RP / 1e4, plan_fee, em),
                "  EEV  = %.2f 元（%.2f 万元）  [点预报计划费 %.2f + 期望紧急费 %.2f]"
                % (EEV, EEV / 1e4, plan_fee_ev, em_ev),
                "  VSS  = EEV − RP = %.2f 元（%.2f 万元）" % (vss, vss / 1e4),
                "  安全裕度（相对点预报计划）= %.2f kWh/日" % rows[-1]['margin'], ""]

    if len(rows) >= 2:
        base, cur = rows[0], rows[-1]
        out += ["---- 负载不确定性的额外代价（σ_L 0 → %.1f%%）----" % (cur['sig'] * 100),
                "  RP 上升 %.2f 万元；EEV 上升 %.2f 万元；VSS 变化 %+.2f 万元"
                % ((cur['RP'] - base['RP']) / 1e4, (cur['EEV'] - base['EEV']) / 1e4,
                   (cur['VSS'] - base['VSS']) / 1e4),
                "  期望紧急费：%.2f → %.2f 万元" % (base['em'] / 1e4, cur['em'] / 1e4)]
    open("_load_uncertainty_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图 ----
    x = [r['sig'] * 100 for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(x, [r['plan_fee'] / 1e4 for r in rows], "o-", label="计划购电费")
    ax[0].plot(x, [r['em'] / 1e4 for r in rows], "s-", label="期望紧急购电费")
    ax[0].plot(x, [r['RP'] / 1e4 for r in rows], "^-", label="RP（期望总费用）")
    ax[0].set_xlabel("负载预报误差标准差 $\\sigma_L$（%）"); ax[0].set_ylabel("万元")
    ax[0].set_title("负载不确定性推高计划费与紧急费"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].bar([str(int(v)) + "%" for v in x], [r['VSS'] / 1e4 for r in rows], color="#16a085")
    for i, r in enumerate(rows):
        ax[1].text(i, r['VSS'] / 1e4 + 1, "%.1f 万" % (r['VSS'] / 1e4), ha="center", fontsize=9)
    ax[1].set_xlabel("负载预报误差 $\\sigma_L$"); ax[1].set_ylabel("VSS（万元）")
    ax[1].set_title("分布感知的价值（VSS）"); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_负载不确定性." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_负载不确定性.png")


if __name__ == "__main__":
    main()
