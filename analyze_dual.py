# -*- coding: utf-8 -*-
"""方向4：对偶 / 影子价格机理分析（LP 的"内功"）。

在问题2 确定性全局 LP（完美预见，连续储能）中，功率平衡不等式
    -G - D + C ≤ (PV - L)·Δt
的对偶变量（scipy 返回 ∂f*/∂RHS，为负）取负号后即"每增加 1 kWh 净供给（多光伏/
少负载）可节省的成本"，即**电力的边际价值（影子价格 λ）**。

结论（用数据验证）：
  λ_t ≤ price_t 恒成立（总能按电价购电，边际价值不超过电价）；
  λ_t = price_t   —— 该时刻靠外网购电（约束紧，边际源=电网）；
  λ_t < price_t   —— 该时刻靠储能放电（搬移了更便宜时段的电量）；
  影子价格与电价之差正是储能套利捕获的价值。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from data_loader import load_fj1, load_fj2, DT, PMAX_E, E_MAX, E_MIN, E_INIT, ETA, N_SLOT

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def _solve_global_with_duals(price_mat, load_mat, pv_mat):
    """复刻 solve_q4._global_lp 的 LP 构造，返回完整 res（含对偶变量）。"""
    ndays = load_mat.shape[0]
    N = ndays * N_SLOT
    Lf = load_mat.ravel() * DT
    PVf = pv_mat.ravel() * DT
    Pf = price_mat.ravel()
    offG, offC, offD, offE = 0, N, 2 * N, 3 * N
    n = 4 * N + 1
    c = np.zeros(n); c[offG:offG + N] = Pf

    rows, cols, vals = [], [], []
    b_eq = []
    for t in range(N):
        rows += [t, t, t, t]
        cols += [offE + t + 1, offE + t, offC + t, offD + t]
        vals += [1.0, -1.0, -ETA, 1.0 / ETA]
        b_eq.append(0.0)
    rows += [N, N + 1]; cols += [offE, offE + N]; vals += [1.0, 1.0]
    b_eq += [E_INIT, E_INIT]
    A_eq = coo_matrix((vals, (rows, cols)), shape=(N + 2, n)).tocsr()

    rows2, cols2, vals2 = [], [], []
    b_ub = []
    for t in range(N):
        rows2 += [t, t, t]; cols2 += [offG + t, offD + t, offC + t]
        vals2 += [-1.0, -1.0, 1.0]
        b_ub.append(PVf[t] - Lf[t])
    A_ub = coo_matrix((vals2, (rows2, cols2)), shape=(N, n)).tocsr()

    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[offC:offC + N] = PMAX_E; ub[offD:offD + N] = PMAX_E
    lb[offE:offE + N + 1] = E_MIN; ub[offE:offE + N + 1] = E_MAX

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res


def main():
    d1 = load_fj1()
    price1 = d1["price"]
    DATES, LOAD, PV = load_fj2()
    price_mat = np.tile(price1, (365, 1))

    res = _solve_global_with_duals(price_mat, LOAD, PV)
    N = 365 * N_SLOT
    lam = -np.asarray(res.ineqlin.marginals).reshape(365, N_SLOT)  # 影子价格 λ≥0
    G = res.x[0:N].reshape(365, N_SLOT)
    D = res.x[2 * N:3 * N].reshape(365, N_SLOT)
    E = res.x[3 * N:3 * N + N + 1]

    price = price_mat
    buying = G > 1e-6                       # 外网购电
    discharging = (D > 1e-6) & (G <= 1e-6)  # 仅储能放电

    out = []
    out.append("===== 方向4 对偶/影子价格机理分析 =====")
    out.append("影子价格 λ（= -功率平衡对偶变量）统计：")
    out.append("  全局 min=%.4f max=%.4f 元/kWh" % (lam.min(), lam.max()))
    out.append("  电价 min=%.4f max=%.4f 元/kWh" % (price.min(), price.max()))
    out.append("  λ ≤ price 恒成立？ %s（max(λ−price)=%.3e）"
               % ("是" if (lam - price).max() < 1e-6 else "否", (lam - price).max()))
    out.append("  外网购电时段 λ≈price：λ 中位数=%.4f，price 中位数=%.4f"
               % (np.median(lam[buying]), np.median(price[buying])))
    out.append("  储能放电时段 λ<price：λ 中位数=%.4f，price 中位数=%.4f"
               % (np.median(lam[discharging]), np.median(price[discharging])))
    out.append("  购电时段占比 %.1f%%，放电时段占比 %.1f%%"
               % (100 * buying.mean(), 100 * discharging.mean()))
    out.append("")
    d_rep = int(np.argmax(PV.sum(axis=1)))
    out.append("代表性一天：第 %d 天（光伏日总量 %s kW）" % (d_rep, PV[d_rep].sum()))
    for tt in (0, 36, 72, 108):
        out.append("  槽%3d (%02d:%02d) price=%.3f λ=%.3f G=%.1f SOC=%.0f"
                   % (tt, tt * 10 // 60, tt * 10 % 60, price[d_rep, tt],
                      lam[d_rep, tt], G[d_rep, tt], E[d_rep * N_SLOT + tt]))
    open("_dual_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图：代表性一天 电价 vs 影子价 vs SOC ----
    h = (np.arange(N_SLOT) + 0.5) * 10 / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    ax[0].step(h, price[d_rep], where="mid", color="#c0392b", lw=1.4, label="电价")
    ax[0].step(h, lam[d_rep], where="mid", color="#16a085", lw=1.4, label="影子价格 λ")
    ax[0].fill_between(h, lam[d_rep], price[d_rep], step="mid",
                       color="#f1c40f", alpha=0.35, label="储能套利差（电价−λ）")
    ax[0].set_ylabel("价格（元/kWh）")
    ax[0].set_title("第 %d 天：电价与电力影子价格" % d_rep)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=0.3)
    Ed = E[d_rep * N_SLOT:(d_rep + 1) * N_SLOT]
    ax[1].plot(h, Ed, color="#2980b9", lw=1.5)
    ax[1].axhline(E_MIN, color="r", ls=":", lw=0.8)
    ax[1].axhline(E_MAX, color="r", ls=":", lw=0.8)
    ax[1].set_ylabel("储电量 SOC（kWh）")
    ax[1].set_xlabel("时刻（h）")
    ax[1].set_title("储电量轨迹（低影子价充电、高影子价放电）")
    ax[1].grid(alpha=0.3)
    for a in ax:
        a.set_xlim(0, 24)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_影子价格." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_影子价格.png")


if __name__ == "__main__":
    main()
