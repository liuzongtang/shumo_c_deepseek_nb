# -*- coding: utf-8 -*-
"""P2 扩展：CVaR 风险厌恶随机规划（在 §6.5 期望型随机规划之上增加风险维度）。

目标：
    min_G   Σ_t p_t·G_t  +  (1/S)Σ_s c_s  +  λ · CVaR_α(c)
其中 c_s = 5·Σ_t p_t·e^s_t 为场景 s 的紧急购电费（元）。
CVaR 用 Rockafellar–Uryasev 线性化：
    CVaR_α(c) = min_θ [ θ + 1/(αS)·Σ_s max(0, c_s − θ) ]
LP 中引入 θ（标量）与 u_s ≥ 0（每场景一个），约束 u_s ≥ c_s − θ，
目标再计入 λ·[θ + 1/(αS)·Σ_s u_s]。

  λ = 0 退化为 §6.5 的期望型随机规划；λ 越大越厌恶"尾部高成本场景"。
  α 取 0.8（S=10 时对应"最坏 2 个场景"的平均），使 CVaR 的样本估计可靠。

口径与 §6.5 完全一致：求解域 = 报告期 2025-02-01~12-31（334 天）；
分段盈余约束在该联合 LP 中不可线性化，故省略 → RP 为下界、VSS 为上界；
所有指标用**精确追索模型** solve_q4._global_recourse 在同一场景集上重估。

用法：python analyze_cvar.py [alpha] [lam1 lam2 ...]
      缺省 alpha=0.8、lam = 0, 0.5, 1.0
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
import analyze_stochastic as A

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def _stoch_plan_cvar(price_mat, load_mat, scenarios, alpha=0.8, lam=0.0):
    """CVaR 风险厌恶两阶段随机规划（联合 LP，无分段盈余约束）。
    变量：G(N) + 每场景块 C(N),D(N),E(N+1),e(N)；末尾 θ(1) 与 u(S)。
    返回 (G (ndays,144), e (S,ndays,144), objective, theta)。"""
    S = scenarios.shape[0]
    ndays = load_mat.shape[0]
    N = ndays * N_SLOT
    Lf = load_mat.ravel() * DT
    Pf = price_mat.ravel()
    offC, offD, offE, offe = 0, N, 2 * N, 3 * N + 1
    blk = 4 * N + 1
    off_theta = N + S * blk
    off_u = off_theta + 1
    n = off_u + S

    c = np.zeros(n)
    c[0:N] = Pf
    for s in range(S):
        b = N + s * blk
        c[b + offe:b + offe + N] = (5.0 / S) * Pf      # 期望紧急购电费
    c[off_theta] = lam                                  # CVaR: θ
    c[off_u:off_u + S] = lam / (alpha * S)              # CVaR: u_s

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

    # ---- 不等式：块1 功率平衡（行 0..S·N−1）；块2 CVaR（行 S·N..S·N+S−1）----
    # b_ub 必须与行号顺序一致，两块分别成批追加。
    rows2, cols2, vals2, b_ub = [], [], [], []
    for s in range(S):                                          # 块1
        b = N + s * blk
        PVe = scenarios[s].ravel() * DT
        base = s * N
        for t in range(N):
            rows2 += [base + t] * 4
            cols2 += [t, b + offD + t, b + offe + t, b + offC + t]
            vals2 += [-1.0, -1.0, -1.0, 1.0]
            b_ub.append(PVe[t] - Lf[t])
    for s in range(S):                                          # 块2：u_s − c_s + θ ≥ 0
        r = S * N + s
        b = N + s * blk
        rows2 += [r] * (N + 2)
        cols2 += [off_u + s] + [b + offe + t for t in range(N)] + [off_theta]
        vals2 += [-1.0] + list(5.0 * Pf) + [-1.0]
        b_ub.append(0.0)
    A_ub = coo_matrix((vals2, (rows2, cols2)), shape=(S * N + S, n)).tocsr()

    lb = np.zeros(n); ub = np.full(n, np.inf)
    for s in range(S):
        b = N + s * blk
        ub[b + offC:b + offC + N] = PMAX_E
        ub[b + offD:b + offD + N] = PMAX_E
        lb[b + offE:b + offE + N + 1] = E_MIN
        ub[b + offE:b + offE + N + 1] = E_MAX
    # θ ≥ 0（c_s ≥ 0 时分位数非负）、u_s ≥ 0 由 lb=0 保证

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    G = res.x[0:N].reshape(ndays, N_SLOT)
    e = np.zeros((S, N))
    for s in range(S):
        b = N + s * blk
        e[s] = res.x[b + offe:b + offe + N]
    return G, e.reshape(S, ndays, N_SLOT), float(res.fun), float(res.x[off_theta])


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.80
    lams = [float(x) for x in sys.argv[2:]] or [0.0, 0.5, 1.0]
    S = 10

    d1 = load_fj1()
    price_all = np.tile(d1["price"], (365, 1))
    DATES, LOAD_all, PV_all = load_fj2()
    pv0_all = build_pv_forecast_stage()[0]
    T0 = A.T0_DAY
    price_mat, LOAD, pv0 = price_all[T0:], LOAD_all[T0:], pv0_all[T0:]
    solve_q4.PRICE_MAT, solve_q4.LOAD = price_all, LOAD_all
    solve_q4.PV, solve_q4.DATES = PV_all, DATES

    err = pv0_all - PV_all
    months = np.array([A._month(d) for d in DATES])
    scen_all = A.make_scenarios(S, pv0_all, err, months)
    scen = scen_all[:, T0:, :]

    # 点预报计划（EV 解）作为对照基准
    G_ev = solve_q4._global_lp(price_mat, LOAD, pv0)['G']
    plan_fee_ev = float(np.sum(G_ev * price_mat))
    em_ev, per_ev = A._recourse_cost_expected(G_ev, price_mat, LOAD, scen)
    k = max(1, int(round((1.0 - alpha) * S)))
    cvar_ev = float(np.sort(per_ev)[::-1][:k].mean())

    out = ["===== CVaR 风险厌恶随机规划（S=%d，α=%.2f，CVaR 取最坏 %d 个场景均值）=====" % (S, alpha, k),
           "口径：报告期 2025-02-01~12-31（334 天）；指标由精确追索模型在同一场景集上重估",
           "λ=0 即 §6.5 的期望型随机规划；λ 越大越厌恶尾部高成本。", "",
           "[对照] 点预报计划（EV 解）：计划费 %.2f 元，期望紧急费 %.2f 元，CVaR=%.2f 元，合计 %.2f 元"
           % (plan_fee_ev, em_ev, cvar_ev, plan_fee_ev + em_ev), ""]

    rows = []
    for lam in lams:
        G, e_lp, obj, theta = _stoch_plan_cvar(price_mat, LOAD, scen, alpha=alpha, lam=lam)
        plan_fee = float(np.sum(G * price_mat))
        em, per = A._recourse_cost_expected(G, price_mat, LOAD, scen)
        cvar = float(np.sort(per)[::-1][:k].mean())
        mean_c = float(np.mean(per))
        rows.append(dict(lam=lam, plan_fee=plan_fee, em=em, cvar=cvar,
                         total=plan_fee + em, risk_adj=plan_fee + em + lam * cvar))
        out += ["---- λ = %.2f ----" % lam,
                "  计划购电费 %.2f + 期望紧急费 %.2f = %.2f 元" % (plan_fee, em, plan_fee + em),
                "  CVaR(%.0f%%) = %.2f 元；风险调整目标 = 计划费+期望紧急费+λ·CVaR = %.2f 元"
                % (alpha * 100, cvar, plan_fee + em + lam * cvar),
                "  安全裕度（相对点预报计划购电增量）= %.2f kWh/日" % float(np.sum(G - G_ev) / G.shape[0]),
                ""]

    out += ["---- λ 权衡 ----",
            "  " + "  ".join("λ=%.2f: 计划费%.1f万 期望紧急%.1f万 CVaR%.1f万"
                             % (r['lam'], r['plan_fee'] / 1e4, r['em'] / 1e4, r['cvar'] / 1e4)
                             for r in rows)]
    if len(rows) >= 2:
        d_plan = rows[-1]['plan_fee'] - rows[0]['plan_fee']
        d_cvar = rows[0]['cvar'] - rows[-1]['cvar']
        out += ["  从 λ=%.2f 到 λ=%.2f：计划费 +%.2f 元，CVaR 下降 %.2f 元（每多付 1 元计划费可压 %.3f 元尾部风险）"
                % (rows[0]['lam'], rows[-1]['lam'], d_plan, d_cvar,
                   (d_cvar / d_plan) if abs(d_plan) > 1e-9 else float('nan'))]
    open("_cvar_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图：λ 权衡 ----
    lam_x = [r['lam'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(lam_x, [r['plan_fee'] / 1e4 for r in rows], "o-", label="计划购电费")
    ax[0].plot(lam_x, [r['em'] / 1e4 for r in rows], "s-", label="期望紧急购电费")
    ax[0].plot(lam_x, [r['total'] / 1e4 for r in rows], "^-", label="期望总费用")
    ax[0].axhline(plan_fee_ev / 1e4, ls=":", color="gray", label="点预报计划费")
    ax[0].set_xlabel(r"风险厌恶权重 $\lambda$"); ax[0].set_ylabel("万元")
    ax[0].set_title("λ 增大：计划费上升、紧急费下降"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot([r['total'] / 1e4 for r in rows], [r['cvar'] / 1e4 for r in rows], "o-", color="#c0392b")
    for r in rows:
        ax[1].annotate(r"$\lambda$=%.2f" % r['lam'], (r['total'] / 1e4, r['cvar'] / 1e4),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_xlabel("期望总费用（万元）"); ax[1].set_ylabel("CVaR（万元）")
    ax[1].set_title("成本—风险权衡前沿"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_CVaR风险权衡." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_CVaR风险权衡.png")


if __name__ == "__main__":
    main()
