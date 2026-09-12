# -*- coding: utf-8 -*-
"""方向4 深化（三项）：对偶分析的"投资信号 + 波动价对比 + 水值期权结构"。

① 容量/功率影子价格（投资信号）：储能容量上界 E_MAX 与功率上界 PMAX 的约束
   对偶变量（变量 reduced cost）直接读出"每多 1 kWh 容量 / 1 kW 功率的年边际价值"，
   并与差分法（重跑 LP）交叉验证；边际价值 = 年化成本 A·c 的交点即最优配置，
   与 §6.7 的 (E*,P*) 精确呼应。
② 波动电价对比：附件4 波动价下做同样的 λ/ψ 分析，对比同价与波动价的储能价值构成
   （净套利 vs 净消纳）、水值范围、容量边际价值。
③ 水值期权结构：储能水值 ψ 与储电量 SOC 的相图（内点平台 + 边界跳变），
   以及 ψ 的月均值季节序列（春夏光伏富余水值低、秋冬紧缺水值高）。

口径：与 §6.3/§6.5 一致，E_MIN=1200、E_INIT=6000 固定，仅扫描 E_MAX/PMAX；
对偶聚合用全年 365 天（对偶变量为全年 LP 之对偶），储能价值构成用报告期 334 天。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import (load_fj1, load_fj2, load_fj4, DT, ETA, E_MIN, E_MAX, E_INIT,
                         N_SLOT)

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

R0 = 31 * N_SLOT          # 报告期起点（2025-02-01 0:00）


def solve(price_mat, load_mat, pv_mat, e_max, p_max, e_min=E_MIN):
    """全年连续储能确定性 LP，返回对偶量。E_MIN/E_INIT 固定，仅 E_MAX/PMAX 可变。

    返回 dict: cost365(全年费), cost334(报告期费), G,C,D,E(365,144)/(*), lam, psi,
               rcC, rcD, rcE（各变量上界 reduced cost，均为 ≤0）。
    """
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
    ub[offC:offC + N] = p_max * DT; ub[offD:offD + N] = p_max * DT
    lb[offE:offE + N + 1] = e_min; ub[offE:offE + N + 1] = e_max

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)

    G = res.x[offG:offG + N].reshape(ndays, N_SLOT)
    C = res.x[offC:offC + N].reshape(ndays, N_SLOT)
    D = res.x[offD:offD + N].reshape(ndays, N_SLOT)
    E = res.x[offE:offE + N + 1]
    lam = -np.asarray(res.ineqlin.marginals).reshape(ndays, N_SLOT)     # 影子价格 λ≥0
    psi = -np.asarray(res.eqlin.marginals)[:N].reshape(ndays, N_SLOT)   # 水值 ψ≥0
    um = np.asarray(res.upper.marginals)                                # 上界 reduced cost ≤0
    rcC = um[offC:offC + N].reshape(ndays, N_SLOT)
    rcD = um[offD:offD + N].reshape(ndays, N_SLOT)
    rcE = um[offE:offE + N + 1]                                         # E 上界（含端点）

    cost365 = float(np.dot(res.x[offG:offG + N], Pf))
    cost334 = float(np.sum(G[31:] * price_mat[31:]))
    return {'G': G, 'C': C, 'D': D, 'E': E, 'lam': lam, 'psi': psi,
            'rcC': rcC, 'rcD': rcD, 'rcE': rcE,
            'cost365': cost365, 'cost334': cost334}


def marginals_of(r):
    """从对偶量聚合出容量/功率的边际价值（元/年，≥0 = 省的钱）。
    MV_E = -Σ rcE（每多 1 kWh 容量）；MV_P = -Σ(rcC+rcD)·DT（每多 1 kW 功率）。"""
    mv_e = -float(r['rcE'].sum())
    mv_p = -float((r['rcC'].sum() + r['rcD'].sum()) * DT)
    return mv_e, mv_p


def storage_value_split(price_mat, load_mat, pv_mat, r):
    """储能价值（报告期 334 天）= 净套利 + 净消纳弃光。返回 (价值, 净套利, 净消纳)。"""
    p = price_mat[31:]
    Lf = load_mat[31:] * DT
    PVf = pv_mat[31:] * DT
    Gf = r['G'][31:]; Cf = r['C'][31:]; Df = r['D'][31:]
    c_no = float(np.sum(p * np.maximum(0.0, Lf - PVf)))
    c_yes = r['cost334']
    value = c_no - c_yes
    arb = float(np.sum(p * (Df - Cf)))                      # 净套利 Σp(D−C)
    residual = PVf - Lf + Gf + Df - Cf                      # 弃光电量
    surplus = np.where(PVf > Lf, PVf - Lf, 0.0)
    absorb = float(np.sum(p * (surplus - residual)))        # 净消纳弃光
    return value, arb, absorb


def main():
    d1 = load_fj1()
    price1 = d1['price']
    _, LOAD, PV = load_fj2()
    _, price4 = load_fj4()
    pm_flat = np.tile(price1, (365, 1))       # 附件1 同价（平铺 365 天）
    ndays = 365

    out = []
    out.append("===== 方向4 深化：对偶的投资信号 + 波动价对比 + 水值期权结构 =====")
    out.append("口径：E_MIN=1200、E_INIT=6000 固定，仅扫描 E_MAX/PMAX；")
    out.append("     对偶聚合用全年 365 天，储能价值构成用报告期 334 天。")
    out.append("")

    # ================= ① 容量/功率影子价格（投资信号） =================
    out.append("【① 容量/功率影子价格：对偶 = 差分（交叉验证）】")
    base = solve(pm_flat, LOAD, PV, E_MAX, 5000.0)
    mv_e0, mv_p0 = marginals_of(base)
    out.append("  对偶容量边际价值  MV_E = %+.4f 元/kWh/年" % mv_e0)
    out.append("  对偶功率边际价值  MV_P = %+.4f 元/kW/年" % mv_p0)
    out.append("  小步长差分（应收敛到对偶值）：")
    for de in (50.0, 100.0, 200.0):
        r = solve(pm_flat, LOAD, PV, E_MAX + de, 5000.0)
        out.append("    E_MAX +%4d kWh → 差分 MV_E = %.4f 元/kWh/年" % (int(de),
                   (base['cost365'] - r['cost365']) / de))
    for dp in (50.0, 100.0, 200.0):
        r = solve(pm_flat, LOAD, PV, E_MAX, 5000.0 + dp)
        out.append("    PMAX +%4d kW  → 差分 MV_P = %.4f 元/kW/年" % (int(dp),
                   (base['cost365'] - r['cost365']) / dp))
    out.append("")

    # 年化成本线
    c_E, c_P = 1000.0, 500.0
    T, rr = 10.0, 0.05
    A = rr / (1 - (1 + rr) ** (-T))
    aE, aP = A * c_E, A * c_P                       # 129.50 元/kWh/年、64.75 元/kW/年
    out.append("  年化边际成本：容量 A·c_E=%.2f 元/kWh/年，功率 A·c_P=%.2f 元/kW/年"
               % (aE, aP))

    # 容量边际价值曲线（固定 P=6000，§6.7 最优功率）
    E_grid = [7200, 9000, 10800, 12600, 14400, 16800, 19200, 21600, 24000, 26400, 28800]
    mvE = []
    for e in E_grid:
        r = solve(pm_flat, LOAD, PV, float(e), 6000.0)
        mvE.append(marginals_of(r)[0])
    out.append("  容量边际价值（P=6000 固定）：")
    for e, m in zip(E_grid, mvE):
        out.append("    E_MAX=%6d → MV_E=%.2f 元/kWh/年" % (e, m))

    # 功率边际价值曲线（固定 E=21600，§6.7 最优容量）
    P_grid = [3000, 4000, 4500, 5000, 5500, 6000, 7000, 8000]
    mvP = []
    for p in P_grid:
        r = solve(pm_flat, LOAD, PV, 21600.0, float(p))
        mvP.append(marginals_of(r)[1])
    out.append("  功率边际价值（E=21600 固定）：")
    for p, m in zip(P_grid, mvP):
        out.append("    PMAX=%5d → MV_P=%.2f 元/kW/年" % (p, m))

    # 交点（线性插值）
    def crossing(xs, ys, level):
        for i in range(len(ys) - 1):
            if (ys[i] - level) * (ys[i + 1] - level) <= 0:
                x0, x1 = xs[i], xs[i + 1]; y0, y1 = ys[i], ys[i + 1]
                return x0 + (x1 - x0) * (level - y0) / (y1 - y0)
        return None
    E_cross = crossing(E_grid, mvE, aE)
    P_cross = crossing(P_grid, mvP, aP)
    out.append("  交点（边际价值=年化成本）：E*≈%.0f kWh、P*≈%.0f kW（对比 §6.7 的 21600/6000）"
               % (E_cross, P_cross))
    out.append("")

    # ================= ② 波动电价对比 =================
    out.append("【② 波动电价（附件4）vs 同价（附件1）：对偶视角】")
    r_flat = solve(pm_flat, LOAD, PV, E_MAX, 5000.0)
    r_vol = solve(price4, LOAD, PV, E_MAX, 5000.0)
    v_f, a_f, ab_f = storage_value_split(pm_flat, LOAD, PV, r_flat)
    v_v, a_v, ab_v = storage_value_split(price4, LOAD, PV, r_vol)
    out.append("  附件4 电价范围 [%.3f, %.3f]（附件1 同价 [%.3f, %.3f]）"
               % (price4.min(), price4.max(), price1.min(), price1.max()))
    out.append("  储能价值（334 天）：同价 %.2f 万 = 净套利 %.2f 万(%4.1f%%) + 净消纳 %.2f 万"
               % (v_f / 1e4, a_f / 1e4, 100 * a_f / v_f, ab_f / 1e4))
    out.append("                    波动价 %.2f 万 = 净套利 %.2f 万(%4.1f%%) + 净消纳 %.2f 万"
               % (v_v / 1e4, a_v / 1e4, 100 * a_v / v_v, ab_v / 1e4))
    out.append("  水值 ψ 范围：同价 [%.4f, %.4f]，波动价 [%.4f, %.4f]"
               % (r_flat['psi'].min(), r_flat['psi'].max(), r_vol['psi'].min(), r_vol['psi'].max()))
    # 波动价下容量边际价值更高（跨日套利更依赖容量）
    mvE_flat = [marginals_of(solve(pm_flat, LOAD, PV, float(e), 6000.0))[0] for e in E_grid]
    mvE_vol = [marginals_of(solve(price4, LOAD, PV, float(e), 6000.0))[0] for e in E_grid]
    out.append("  容量边际价值 MV_E（P=6000）：")
    for e, mf, mv in zip(E_grid, mvE_flat, mvE_vol):
        out.append("    E_MAX=%6d → 同价 %.2f / 波动价 %.2f 元/kWh/年" % (e, mf, mv))
    out.append("")

    # ================= ③ 水值期权结构 =================
    out.append("【③ 储能水值 ψ 的蓄水—兑现结构（同价，报告期 334 天）】")
    sl = slice(31, ndays)
    psi = r_flat['psi'][sl].ravel()               # 每段水值 (334*144,)
    Esoc = r_flat['E'][R0:R0 + (ndays - 31) * N_SLOT]  # 每段起始电量
    corr_ep = float(np.corrcoef(Esoc, psi)[0, 1])
    out.append("  ψ 跟随电价相位（非 SOC 库存）：ψ∈[%.4f, %.4f]，ψ−SOC 相关系数 %+.2f"
               % (psi.min(), psi.max(), corr_ep))
    # 分段：SOC 内点 vs 边界
    interior = (Esoc > E_MIN + 1e-3) & (Esoc < E_MAX - 1e-3)
    out.append("  SOC 内点时段 %d 个（%.1f%%），ψ 均值 %.4f；"
               % (interior.sum(), 100 * interior.mean(), psi[interior].mean()))
    out.append("  SOC 触顶（满电待放）/触底（空电待蓄）时段 %d/%d 个，ψ 均值 %.4f / %.4f"
               % ((Esoc >= E_MAX - 1e-3).sum(), (Esoc <= E_MIN + 1e-3).sum(),
                  psi[Esoc >= E_MAX - 1e-3].mean() if (Esoc >= E_MAX - 1e-3).any() else 0,
                  psi[Esoc <= E_MIN + 1e-3].mean() if (Esoc <= E_MIN + 1e-3).any() else 0))
    out.append("  → 满电（白天光伏充满、临近峰价兑现）水值高，空电（深夜/清晨、谷价蓄水）水值低。")
    # 月均值（2025-02 ~ 12，334 天）
    month_edges = [0, 28, 59, 89, 120, 150, 181, 211, 242, 272, 303, 334]   # 相对 2/1 的天数
    month_names = ['2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    psi2d = psi.reshape(ndays - 31, N_SLOT)
    psi_month = []
    for i in range(11):
        psi_month.append(psi2d[month_edges[i]:month_edges[i + 1]].mean())
    out.append("  ψ 月均值（元/kWh）：" + "  ".join("%s=%.3f" % (m, v)
               for m, v in zip(month_names, psi_month)))
    out.append("")
    open("_dual_ext_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ================= 图 15：容量/功率边际价值（投资信号） =================
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(E_grid, mvE, "o-", color="#2980b9", lw=2, label="容量边际价值 MV_E")
    ax[0].axhline(aE, color="#c0392b", lw=1.5, ls="--", label="年化容量成本 A·c_E=%.0f" % aE)
    ax[0].axvline(E_cross, color="k", lw=0.8, ls=":")
    ax[0].scatter([E_cross], [aE], marker="*", s=220, color="k", zorder=5,
                  label="最优容量 E*≈%.0f" % E_cross)
    ax[0].set_xlabel("储能容量 E_MAX（kWh）")
    ax[0].set_ylabel("年边际价值（元/kWh/年）")
    ax[0].set_title("(a) 容量：增容至边际价值=年化成本")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(P_grid, mvP, "o-", color="#16a085", lw=2, label="功率边际价值 MV_P")
    ax[1].axhline(aP, color="#c0392b", lw=1.5, ls="--", label="年化功率成本 A·c_P=%.0f" % aP)
    ax[1].axvline(P_cross, color="k", lw=0.8, ls=":")
    ax[1].scatter([P_cross], [aP], marker="*", s=220, color="k", zorder=5,
                  label="最优功率 P*≈%.0f" % P_cross)
    ax[1].set_xlabel("储能功率 PMAX（kW）")
    ax[1].set_ylabel("年边际价值（元/kW/年）")
    ax[1].set_title("(b) 功率：增功率至边际价值=年化成本")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_影子价格_投资信号." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ================= 图 16：波动价 vs 同价 =================
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(2)
    w = 0.38
    ax[0].bar(x - w / 2, [a_f / 1e4, a_v / 1e4], w, color="#e67e22", label="净套利（峰谷搬移）")
    ax[0].bar(x + w / 2, [ab_f / 1e4, ab_v / 1e4], w, color="#27ae60", label="净消纳弃光")
    for xi, (v1, v2) in enumerate(zip([a_f / 1e4, ab_f / 1e4], [a_v / 1e4, ab_v / 1e4])):
        pass
    for xi in range(2):
        tot = (a_f + ab_f) / 1e4 if xi == 0 else (a_v + ab_v) / 1e4
        ax[0].text(xi, tot + 2, "%.1f" % tot, ha="center", fontsize=9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["附件1 同价", "附件4 波动价"])
    ax[0].set_ylabel("储能价值构成（万元）")
    ax[0].set_title("(a) 波动价抬高储能价值（跨日套利）")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].plot(E_grid, mvE_flat, "o-", color="#2980b9", lw=2, label="同价")
    ax[1].plot(E_grid, mvE_vol, "s-", color="#e67e22", lw=2, label="波动价")
    ax[1].axhline(aE, color="#c0392b", lw=1.5, ls="--", label="年化容量成本 A·c_E")
    ax[1].set_xlabel("储能容量 E_MAX（kWh）")
    ax[1].set_ylabel("容量边际价值（元/kWh/年）")
    ax[1].set_title("(b) 波动价下容量边际价值更高")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_影子价格_波动价对比." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ================= 图 17：水值期权结构 =================
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].scatter(Esoc, psi, s=3, alpha=0.3, color="#8e44ad")
    ax[0].axvline(E_MIN, color="r", ls=":", lw=0.9)
    ax[0].axvline(E_MAX, color="r", ls=":", lw=0.9)
    ax[0].set_xlabel("储电量 SOC（kWh）")
    ax[0].set_ylabel("水值 ψ（元/kWh）")
    ax[0].set_title("(a) ψ−SOC 相图：满电待放高水值、空电待蓄低水值")
    ax[0].grid(alpha=0.3)

    ax[1].plot(month_names, psi_month, "o-", color="#2980b9", lw=2)
    ax[1].set_xlabel("月份（2025 报告期）")
    ax[1].set_ylabel("水值 ψ 月均值（元/kWh）")
    ax[1].set_title("(b) 水值季节结构：春夏低、秋冬高")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_水值期权结构." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(out))
    print("已生成 figures/fig_影子价格_投资信号.png / fig_影子价格_波动价对比.png / fig_水值期权结构.png")


if __name__ == "__main__":
    main()
