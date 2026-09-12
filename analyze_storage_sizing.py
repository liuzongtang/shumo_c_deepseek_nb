# -*- coding: utf-8 -*-
"""储能容量/功率最优配置（投资规划）。

框架（确定性、完美预见下运行）：
  V(E,P) = 全年"无储能"购电费 - 全年"有储能(E,P)"购电费   （储能年价值，元）
  无储能购电费 = Σ_t p_t·max(0, L_t-PV_t)·Δt
  有储能购电费 = 全年连续储能 LP（SOC∈[0.1E,E]，功率≤P）
  年化投资成本 = A·(c_E·E + c_P·P)，A = r/(1-(1+r)^-T) 为等额年金因子
  净年收益  NB(E,P) = V(E,P) - 年化投资成本
在 (E,P) 网格上求 NB 最大者 (E*,P*)，并对单位成本做敏感性。
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

from data_loader import load_fj1, load_fj2, N_SLOT, DT

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def _det_lp(price_mat, load_mat, pv_mat, e_max, p_max):
    """全年连续储能确定性 LP，容量 e_max(kWh)、功率 p_max(kW)。
    返回报告期(2.1-12.31, 334天)购电费(元)。SOC 下限 e_max/9、初末电量取中值
    (与附件1 E_MIN=1200=E_MAX/9、E_INIT=6000=(E_MIN+E_MAX)/2 一致)。"""
    ndays = load_mat.shape[0]
    N = ndays * N_SLOT
    Lf = load_mat.ravel() * DT
    PVf = pv_mat.ravel() * DT
    Pf = price_mat.ravel()
    e_min = e_max / 9.0
    p_e = p_max * DT                 # 每段最大充/放电量 kWh
    e_init = (e_min + e_max) / 2.0

    offG, offC, offD, offE = 0, N, 2 * N, 3 * N
    n = 4 * N + 1
    c = np.zeros(n); c[offG:offG + N] = Pf

    rows, cols, vals = [], [], []
    b_eq = []
    for t in range(N):
        rows += [t, t, t, t]
        cols += [offE + t + 1, offE + t, offC + t, offD + t]
        vals += [1.0, -1.0, -0.9, 1.0 / 0.9]
        b_eq.append(0.0)
    rows += [N, N + 1]; cols += [offE, offE + N]; vals += [1.0, 1.0]
    b_eq += [e_init, e_init]
    A_eq = coo_matrix((vals, (rows, cols)), shape=(N + 2, n)).tocsr()

    rows2, cols2, vals2 = [], [], []
    b_ub = []
    for t in range(N):
        rows2 += [t, t, t]; cols2 += [offG + t, offD + t, offC + t]
        vals2 += [-1.0, -1.0, 1.0]
        b_ub.append(PVf[t] - Lf[t])
    A_ub = coo_matrix((vals2, (rows2, cols2)), shape=(N, n)).tocsr()

    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[offC:offC + N] = p_e; ub[offD:offD + N] = p_e
    lb[offE:offE + N + 1] = e_min; ub[offE:offE + N + 1] = e_max

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    G = res.x[offG:offG + N].reshape(ndays, N_SLOT)
    sl = slice(31, ndays)                    # 报告期 334 天
    return float(np.sum(G[sl] * price_mat[sl]))


def _no_storage_cost(price_mat, load_mat, pv_mat):
    """无储能：逐段买入实际缺口，弃光。返回报告期(334天)购电费(元)。"""
    deficit = np.maximum(0.0, load_mat - pv_mat) * DT
    sl = slice(31, load_mat.shape[0])
    return float(np.sum(deficit[sl] * price_mat[sl]))


def main():
    d1 = load_fj1()
    price1 = d1['price']
    _, load, pv = load_fj2()
    price_mat = np.tile(price1, (365, 1))

    c_no = _no_storage_cost(price_mat, load, pv)

    # 成本参数
    c_E = 1000.0     # 元/kWh 容量成本
    c_P = 500.0      # 元/kW  功率成本
    T, r = 10.0, 0.05
    A = r / (1 - (1 + r) ** (-T))     # 年金因子 ≈ 0.1295

    E_grid = [4000.0, 6000.0, 8000.0, 10000.0, 10800.0, 12000.0, 14400.0,
              18000.0, 21600.0, 24000.0, 28800.0]
    P_grid = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0]

    V = np.zeros((len(E_grid), len(P_grid)))
    for i, e in enumerate(E_grid):
        for j, p in enumerate(P_grid):
            V[i, j] = c_no - _det_lp(price_mat, load, pv, e, p)

    invest = A * (c_E * np.array(E_grid)[:, None] + c_P * np.array(P_grid)[None, :])
    NB = V - invest

    i_cur = E_grid.index(10800.0)
    j_cur = P_grid.index(5000.0)
    out = []
    out.append("===== 储能容量/功率最优配置（投资规划） =====")
    out.append("无储能全年购电费 = %.2f 元（%.2f 万元）" % (c_no, c_no / 1e4))
    out.append("当前配置 E=10800kWh,P=5000kW 的储能价值 = %.2f 万元" % (V[i_cur, j_cur] / 1e4))
    out.append("成本参数：c_E=%.0f 元/kWh, c_P=%.0f 元/kW, T=%.0f 年, r=%.0f%%, 年金因子 A=%.4f"
               % (c_E, c_P, T, r * 100, A))
    out.append("")

    ib, jb = np.unravel_index(np.argmax(NB), NB.shape)
    E_star, P_star = E_grid[ib], P_grid[jb]
    out.append("净年收益最大：E*=%.0f kWh, P*=%.0f kW" % (E_star, P_star))
    out.append("  储能价值 V=%.2f 万元, 年化投资=%.2f 万元, 净收益 NB=%.2f 万元"
               % (V[ib, jb] / 1e4, invest[ib, jb] / 1e4, NB[ib, jb] / 1e4))
    out.append("")

    out.append("E×P 网格净年收益 NB（万元），粗体为最大：")
    out.append("         " + "".join("%9d" % int(p) for p in P_grid))
    for i, e in enumerate(E_grid):
        line = "E=%6d " % int(e)
        for j in range(len(P_grid)):
            mark = "*" if (i == ib and j == jb) else " "
            line += "%8.1f%s" % (NB[i, j] / 1e4, mark)
        out.append(line)
    out.append("")

    # 敏感性：不同单位容量成本下的最优 (E*,P*)
    out.append("单位容量成本 c_E 敏感性（c_P=500 元/kW 固定）：")
    for cE in (600.0, 800.0, 1000.0, 1200.0, 1500.0):
        cP = 500.0
        inv = A * (cE * np.array(E_grid)[:, None] + cP * np.array(P_grid)[None, :])
        nb = V - inv
        ib, jb = np.unravel_index(np.argmax(nb), nb.shape)
        out.append("  c_E=%4.0f 元/kWh -> E*=%6.0f kWh, P*=%5.0f kW, NB*=%.1f 万元"
                   % (cE, E_grid[ib], P_grid[jb], nb[ib, jb] / 1e4))

    open("_storage_sizing_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图：价值面 + 净收益面 + 最优标记 ----
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    Pg, Eg = np.meshgrid(P_grid, E_grid)
    c1 = ax[0].contourf(Pg, Eg, V / 1e4, levels=12, cmap="YlGnBu")
    fig.colorbar(c1, ax=ax[0], label="储能年价值（万元）")
    ax[0].scatter([5000], [10800], marker="o", s=90, facecolors="none",
                  edgecolors="red", lw=2, label="当前配置(10800,5000)")
    ax[0].set_xlabel("功率 P (kW)")
    ax[0].set_ylabel("容量 E (kWh)")
    ax[0].set_title("(a) 储能年价值 V(E,P)")
    ax[0].legend(fontsize=8)

    c2 = ax[1].contourf(Pg, Eg, NB / 1e4, levels=12, cmap="RdYlGn")
    fig.colorbar(c2, ax=ax[1], label="净年收益（万元）")
    ax[1].scatter([P_star], [E_star], marker="*", s=200, color="k",
                  label="最优 (%.0f, %.0f)" % (P_star, E_star))
    ax[1].scatter([5000], [10800], marker="o", s=90, facecolors="none",
                  edgecolors="red", lw=2, label="当前配置")
    ax[1].set_xlabel("功率 P (kW)")
    ax[1].set_ylabel("容量 E (kWh)")
    ax[1].set_title("(b) 净年收益 NB(E,P)（c_E=%.0f 元/kWh）" % c_E)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_储能最优配置." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(out))
    print("已生成 figures/fig_储能最优配置.png")


if __name__ == "__main__":
    main()
