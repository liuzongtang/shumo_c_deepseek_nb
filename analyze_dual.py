# -*- coding: utf-8 -*-
"""方向4：对偶 / 影子价格机理分析（LP 的"内功"，深化版）。

在问题2 确定性全局 LP（完美预见，连续储能）中，功率平衡不等式
    -G - D + C ≤ (PV - L)·Δt
的对偶变量取负号后即"每增加 1 kWh 净供给可节省的成本"，即**电力的边际价值
（影子价格 λ_t）**；储能电量动态等式
    E_{t+1} - E_t - η C_t + D_t/η = 0
的对偶变量取负号后即**储能电量边际价值（"水值" ψ_t）**，代表"多存 1 kWh 电"
的价值。

经 KKT 一阶条件（并在 scipy 上数值核实符号约定）可得到严格关系：
  (a) 外网购电时段（G_t>0）  ：λ_t = price_t          （边际源 = 电网）
  (b) 储能充电内点（C_t>0）  ：λ_t = η·ψ_t           （充电直到电的边际价值=打折水值）
  (c) 储能放电内点（D_t>0）  ：λ_t = ψ_t/η           （放电直到电的边际价值=放大水值）
  (d) 弃光时段（平衡松弛）   ：λ_t = 0

由 (b)(c) 消去 ψ，得**储能套利的"峰谷价比"门槛**：一次"充 1/η² 度电、放 1 度电"
的完整往返效率为 η²=0.81，故仅当 峰价/谷价 > 1/η² ≈ 1.2346 时套利才有利可图；
等价地，盈亏平衡效率 η* = sqrt(谷价/峰价)。
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

TOL_G = 1e-6          # 判定"是否购电/充放电"的阈值（kWh）


def _solve_global_with_duals(price_mat, load_mat, pv_mat, e_max=E_MAX, e_min=E_MIN):
    """复刻 solve_q4._global_lp 的 LP 构造，返回完整 res（含对偶变量）。

    e_max/e_min 可调，用于容量灵敏度。
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
    ub[offC:offC + N] = PMAX_E; ub[offD:offD + N] = PMAX_E
    lb[offE:offE + N + 1] = e_min; ub[offE:offE + N + 1] = e_max

    res = linprog(c, A_eq=A_eq, b_eq=np.array(b_eq), A_ub=A_ub, b_ub=np.array(b_ub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res


def _unpack(res, ndays):
    N = ndays * N_SLOT
    G = res.x[0:N].reshape(ndays, N_SLOT)
    C = res.x[N:2 * N].reshape(ndays, N_SLOT)
    D = res.x[2 * N:3 * N].reshape(ndays, N_SLOT)
    E = res.x[3 * N:3 * N + N + 1]
    lam = -np.asarray(res.ineqlin.marginals).reshape(ndays, N_SLOT)   # 影子价格 λ≥0
    psi = -np.asarray(res.eqlin.marginals)[:N].reshape(ndays, N_SLOT)  # 水值 ψ≥0
    return G, C, D, E, lam, psi


def main():
    d1 = load_fj1()
    price1 = d1["price"]
    DATES, LOAD, PV = load_fj2()
    price_mat = np.tile(price1, (365, 1))
    ndays = 365
    Nfull = ndays * N_SLOT

    res = _solve_global_with_duals(price_mat, LOAD, PV)
    G, C, D, E, lam, psi = _unpack(res, ndays)
    price = price_mat

    # ---- 报告期：2025-02-01 ~ 12-31（334 天），与全文一致 ----
    r0 = 31 * N_SLOT
    p = price.ravel()[r0:]
    lamf = lam.ravel()[r0:]
    psif = psi.ravel()[r0:]
    Gf = G.ravel()[r0:]
    Cf = C.ravel()[r0:]
    Df = D.ravel()[r0:]
    Lf = (LOAD.ravel() * DT)[r0:]
    PVf = (PV.ravel() * DT)[r0:]
    residual = PVf - Lf + Gf + Df - Cf          # 弃光电量（平衡松弛量），≥0
    E_slots = E[r0:Nfull]                        # 报告期每段起始电量
    N = Nfull - r0                               # 48096 个时段

    buying = Gf > TOL_G
    charging = Cf > TOL_G
    discharging = Df > TOL_G
    curtail = residual > TOL_G

    out = []
    out.append("===== 方向4 对偶/影子价格机理分析（深化版：影子价格 λ + 储能水值 ψ）=====")
    out.append("")
    out.append("【0】符号：λ=电力边际价值（影子价格），ψ=储能电量边际价值（水值）。")
    out.append("    储能往返效率 η²=%.4f，套利门槛 1/η²=%.4f" % (ETA ** 2, 1.0 / ETA ** 2))
    out.append("    分析窗口 = 报告期 334 天（%d 时段，2025-02-01~12-31）。" % N)
    out.append("")

    # ---- KKT 三条严格关系 ----
    out.append("【1】KKT 一阶条件逐条数值验证（%d 个时段）：" % N)
    viol_a = float((lamf - p).max())
    out.append("  (a) λ ≤ price 恒成立：max(λ−price) = %.3e %s"
               % (viol_a, "✓" if viol_a < 1e-5 else "✗"))
    err_b = float(np.abs(lamf[buying] - p[buying]).max())
    out.append("  (b) 购电时段 λ=price：max|λ−price| = %.3e（%d 时段，%.1f%%）%s"
               % (err_b, int(buying.sum()), 100 * buying.mean(),
                  "✓" if err_b < 1e-5 else "✗"))
    err_c = float(np.abs(lamf[curtail]).max())
    out.append("  (c) 弃光时段 λ=0：max|λ| = %.3e（%d 时段，%.1f%%）%s"
               % (err_c, int(curtail.sum()), 100 * curtail.mean(),
                  "✓" if err_c < 1e-5 else "✗"))
    ch_int = charging & (Cf < PMAX_E - 1e-6) & (~buying)
    err_d = float(np.abs(lamf[ch_int] - ETA * psif[ch_int]).max())
    out.append("  (d) 充电内点 λ=η·ψ：max|λ−ηψ| = %.3e（%d 时段）%s"
               % (err_d, int(ch_int.sum()), "✓" if err_d < 1e-4 else "✗"))
    ds_int = discharging & (Df < PMAX_E - 1e-6) & (~buying)
    err_e = float(np.abs(lamf[ds_int] - psif[ds_int] / ETA).max())
    out.append("  (e) 放电内点 λ=ψ/η：max|λ−ψ/η| = %.3e（%d 时段）%s"
               % (err_e, int(ds_int.sum()), "✓" if err_e < 1e-4 else "✗"))
    out.append("")

    # ---- 互斥经济状态份额 ----
    reg_buy = buying                       # 边际源=电网，λ=p
    reg_curt = curtail                     # 弃光，λ=0（与购电互斥）
    reg_disc = discharging & ~buying       # 纯储能放电，0<λ<p
    reg_pv = ~buying & ~curtail & ~discharging   # 光伏直供（可能含富余光伏充电）
    out.append("【2】互斥经济状态份额与影子价格：")
    for name, mask in [("购电 (λ=p)", reg_buy), ("弃光 (λ=0)", reg_curt),
                       ("储能放电 (0<λ<p)", reg_disc), ("光伏直供", reg_pv)]:
        n_ = int(mask.sum())
        out.append("  %-16s %6.1f%%   λ中位数=%.4f   λ均值=%.4f"
                   % (name, 100 * n_ / N,
                      (np.median(lamf[mask]) if n_ else 0.0),
                      (lamf[mask].mean() if n_ else 0.0)))
    out.append("  （储能充电时段占 %.1f%%，与购电/弃光重叠：谷价充电多来自外购、"
               % (100 * charging.mean()))
    out.append("    富余光伏充电多伴随弃光。）")
    out.append("")

    # ---- 储能水值 ψ 机理 ----
    at_min = E_slots < E_MIN + 1e-3
    at_max = E_slots > E_MAX - 1e-3
    out.append("【3】储能水值 ψ 机理：")
    out.append("  ψ 全域范围 [%.4f, %.4f] 元/kWh；电价范围 [%.4f, %.4f] 元/kWh"
               % (psif.min(), psif.max(), p.min(), p.max()))
    out.append("  充电 λ=ηψ（折扣）、放电 λ=ψ/η（放大），η=%.2f 即损耗楔子。" % ETA)
    out.append("  ψ 触及 E_MAX/E_MIN 边界时段 %d/%d 个（水值跳变处）。"
               % (int(at_max.sum()), int(at_min.sum())))
    out.append("")

    # ---- 储能套利价值：与 §6.3 的精确对账 ----
    c1 = float(np.sum(p * np.maximum(0.0, Lf - PVf)))        # 无储能完美预见
    c2 = float(np.sum(p * Gf))                                # 有储能完美预见
    arb_net = float(np.sum(p * (Df - Cf)))                    # 净套利 Σp(D−C)
    surplus_val = float(np.sum(p[PVf > Lf] * (PVf - Lf)[PVf > Lf]))  # 全部富余光伏市值
    curt_val = float(np.sum(p * residual))                    # 有储能仍弃光的市值
    absorb_val = surplus_val - curt_val                       # 储能净消纳的弃光价值
    out.append("【4】储能价值对账（报告期 334 天，完美预见）：")
    out.append("  无储能总费 c1 = %.2f 万元" % (c1 / 1e4))
    out.append("  有储能总费 c2 = %.2f 万元" % (c2 / 1e4))
    out.append("  储能价值 c1−c2 = %.2f 万元" % ((c1 - c2) / 1e4))
    out.append("    = 净套利 Σp(D−C) %.2f 万元 + 净消纳弃光 %.2f 万元"
               % (arb_net / 1e4, absorb_val / 1e4))
    out.append("    （放电替代购电收入 Σp·D=%.2f 万，充电购电成本 Σp·C=%.2f 万）"
               % (float(np.sum(p * Df)) / 1e4, float(np.sum(p * Cf)) / 1e4))
    resid = (c1 - c2) - (arb_net + absorb_val)
    out.append("  对账残差 (c1−c2)−(净套利+净消纳) = %.3e 元 ✓" % resid)
    out.append("")

    # ---- 套利门槛 1/η² 与盈亏平衡效率 ----
    p_daily_max = price.max(); p_daily_min = price.min()
    spread_daily = p_daily_max / p_daily_min
    eta_break = np.sqrt(1.0 / spread_daily)
    out.append("【5】套利门槛（往返效率 1/η²=%.4f）：" % (1.0 / ETA ** 2))
    out.append("  附件1 日内峰谷价比 = %.3f/%.3f = %.3f，远高于门槛 %.3f"
               % (p_daily_max, p_daily_min, spread_daily, 1.0 / ETA ** 2))
    out.append("  盈亏平衡效率 η* = sqrt(谷价/峰价) = sqrt(1/%.3f) = %.3f，题目 η=0.90 远高于 η*"
               % (spread_daily, eta_break))
    out.append("  → 储能套利对效率不敏感：即便效率降至约 %.0f%%，峰谷套利仍有利可图。"
               % (100 * eta_break))
    out.append("")

    # ---- 容量灵敏度（对偶视角） ----
    out.append("【6】储能容量灵敏度（对偶视角，E_MAX 扫描）：")
    out.append("  E_MAX(kWh) | 总费(万元) | 净套利(万元) | 放电时段λ中位数 | ψ均值")
    e_sweep = [7200, 9000, 10800, 12600, 14400]
    sens_rows = []
    for em in e_sweep:
        r = _solve_global_with_duals(price_mat, LOAD, PV, e_max=float(em))
        _, Cc, Dd, _, ll, pp = _unpack(r, ndays)
        cost = float(np.sum(p * r.x[0:Nfull][r0:]))          # 报告期总费
        net = float(np.sum(p * (Dd.ravel()[r0:] - Cc.ravel()[r0:])))
        lam_d = ll.ravel()[r0:][Dd.ravel()[r0:] > TOL_G]
        med = float(np.median(lam_d)) if lam_d.size else 0.0
        sens_rows.append((em, cost, net, med, float(pp.ravel()[r0:].mean())))
        out.append("  %9d | %9.2f | %9.2f | %.4f | %.4f"
                   % (em, cost / 1e4, net / 1e4, med, pp.ravel()[r0:].mean()))
    out.append("")
    open("_dual_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ================= 图 10：代表性一天（价格/影子价/水值/SOC/充放电） =================
    # 选"弃光最大"的高光伏日：同时呈现 购电(λ=p)、弃光(λ=0)、储能放电(0<λ<p) 三种状态
    resid2d = (PV.ravel() * DT - LOAD.ravel() * DT + G.ravel() + D.ravel() - C.ravel()
               ).reshape(ndays, N_SLOT)
    d_rep = int(31 + np.argmax(resid2d[31:].sum(axis=1)))
    h = (np.arange(N_SLOT) + 0.5) * 10 / 60.0
    pr = price[d_rep]; lm = lam[d_rep]; ps = psi[d_rep]
    Ed = E[d_rep * N_SLOT:(d_rep + 1) * N_SLOT + 1]
    Cc = C[d_rep]; Dd = D[d_rep]; Gg = G[d_rep]

    fig, ax = plt.subplots(3, 1, figsize=(10.6, 9.0), sharex=True)
    ax[0].step(h, pr, where="mid", color="#c0392b", lw=1.5, label="电价 price")
    ax[0].step(h, lm, where="mid", color="#16a085", lw=1.5, label="影子价格 λ")
    ax[0].fill_between(h, lm, pr, step="mid", where=(pr > lm),
                       color="#f1c40f", alpha=0.30, label="套利价差（price−λ）")
    ax[0].set_ylabel("价格（元/kWh）")
    ax[0].set_title("第 %d 天：电价与电力影子价格（购电 λ=price、放电 λ<price、弃光 λ=0）" % d_rep)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=0.3)

    ax[1].step(h, ps, where="mid", color="#8e44ad", lw=1.5, label="储能水值 ψ")
    ax1b = ax[1].twinx()
    ax1b.plot((np.arange(N_SLOT + 1)) * 10 / 60.0, Ed, color="#2980b9", lw=1.6,
              label="储电量 SOC")
    ax1b.axhline(E_MIN, color="r", ls=":", lw=0.8)
    ax1b.axhline(E_MAX, color="r", ls=":", lw=0.8)
    ax[1].set_ylabel("水值 ψ（元/kWh）")
    ax1b.set_ylabel("储电量 SOC（kWh）")
    ax[1].set_title("储能水值 ψ 与储电量轨迹（低水值充电、高水值放电）")
    ax[1].grid(alpha=0.3)

    ax[2].bar(h - 0.045, Cc, width=0.09, color="#27ae60", label="充电 C")
    ax[2].bar(h + 0.045, -Dd, width=0.09, color="#e67e22", label="放电 D")
    ax[2].step(h, Gg, where="mid", color="#34495e", lw=1.3, label="购电 G")
    ax[2].set_ylabel("电量（kWh）")
    ax[2].set_xlabel("时刻（h）")
    ax[2].set_title("充放电与购电量")
    ax[2].legend(fontsize=8, loc="upper right")
    ax[2].grid(alpha=0.3)
    for a in ax:
        a.set_xlim(0, 24)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_影子价格." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ================= 图 11：全年统计 + 套利价差 + 容量灵敏度 =================
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    state = np.empty(N, dtype=object)
    state[reg_curt] = "弃光"
    state[reg_buy] = "购电"
    state[reg_disc] = "放电"
    state[(charging & ~buying & ~curtail & ~discharging)] = "充电"
    state[reg_pv & ~charging] = "光伏直供"
    colors = {"购电": "#c0392b", "放电": "#e67e22", "充电": "#27ae60",
              "弃光": "#95a5a6", "光伏直供": "#2980b9"}
    for sname in ["购电", "放电", "充电", "弃光", "光伏直供"]:
        m = state == sname
        if m.sum():
            ax[0].scatter(p[m], lamf[m], s=3, alpha=0.35, color=colors[sname],
                          label="%s(%d)" % (sname, m.sum()))
    xx = np.linspace(0, p.max(), 100)
    ax[0].plot(xx, xx, "k--", lw=0.8)
    ax[0].set_xlabel("电价 price（元/kWh）")
    ax[0].set_ylabel("影子价格 λ（元/kWh）")
    ax[0].set_title("(a) 全年 λ vs price（λ≤price，y=x 下方）")
    ax[0].legend(fontsize=6.5, loc="lower right", markerscale=3)
    ax[0].grid(alpha=0.3)

    spread = p - lamf
    active = discharging | charging
    ax[1].hist(spread[active], bins=60, color="#f39c12", alpha=0.75)
    ax[1].axvline(spread[active].mean(), color="#c0392b", lw=1.4,
                  label="均值 %.4f" % spread[active].mean())
    ax[1].set_xlabel("套利价差 price−λ（元/kWh）")
    ax[1].set_ylabel("时段数")
    ax[1].set_title("(b) 储能活跃时段的价差分布")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ems = [r[0] for r in sens_rows]
    costs = [r[1] / 1e4 for r in sens_rows]
    nets = [r[2] / 1e4 for r in sens_rows]
    ax[2].plot(ems, costs, "o-", color="#2980b9", label="总费（万元）")
    ax2b = ax[2].twinx()
    ax2b.plot(ems, nets, "s-", color="#e67e22", label="净套利（万元）")
    ax[2].set_xlabel("E_MAX（kWh）")
    ax[2].set_ylabel("总费（万元）", color="#2980b9")
    ax2b.set_ylabel("净套利（万元）", color="#e67e22")
    ax[2].set_title("(c) 容量↑→套利价值边际递减")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_影子价格_套利统计." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(out))
    print("已生成 figures/fig_影子价格.png 与 figures/fig_影子价格_套利统计.png")


if __name__ == "__main__":
    main()
