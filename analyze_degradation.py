# -*- coding: utf-8 -*-
"""P2 扩展：储能循环寿命（老化）成本对调度策略的影响。

目标函数加入**储能吞吐老化成本**（度电成本模型）：
    Total = Σ_t p_t·G_t  +  c_deg · Σ_t D_t
其中 c_deg（元/kWh）为"每 kWh 放电量的等效老化成本"；c_deg = 0 退化为 §5.2 的
纯购电费最小化。该项是**线性**的，因此不改变 LP 结构，只改变最优调度的权衡：
储能套利收益需覆盖"价差 − 老化成本"才值得进行。

口径：报告期 2025-02-01~12-31（334 天），全年连续储能、首末 6000 kWh、SOC∈[1200,10800]；
分别评估两种计划依据：**确定性（完美预见实际光伏，理论下界）** 与 **点预报（0:00 预报）**。

用法：python analyze_degradation.py [c1 c2 ...]     缺省 0 0.1 0.2 0.3 0.5
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
T0_DAY = 31


def _global_lp_deg(price_mat, load_mat, pv_mat, c_deg=0.0):
    """全年连续储能 LP：min Σp·G + c_deg·Σ D。
    返回 dict(G,C,D,E, energy_cost, deg_cost, total)。
    结构与 solve_q4._global_lp 完全一致，仅目标多一项 c_deg·D。"""
    ndays = load_mat.shape[0]
    N = ndays * N_SLOT
    Lf = load_mat.ravel() * DT
    PVf = pv_mat.ravel() * DT
    Pf = price_mat.ravel()
    offG, offC, offD, offE = 0, N, 2 * N, 3 * N
    n = 4 * N + 1

    c = np.zeros(n)
    c[offG:offG + N] = Pf
    if c_deg:
        c[offD:offD + N] = c_deg                      # 储能吞吐老化成本

    rows, cols, vals, b_eq = [], [], [], []
    for t in range(N):                                # SOC 动态
        rows += [t, t, t, t]
        cols += [offE + t + 1, offE + t, offC + t, offD + t]
        vals += [1.0, -1.0, -ETA, 1.0 / ETA]
        b_eq.append(0.0)
    rows += [N, N + 1]; cols += [offE, offE + N]; vals += [1.0, 1.0]
    b_eq += [E_INIT, E_INIT]
    A_eq = coo_matrix((vals, (rows, cols)), shape=(N + 2, n)).tocsr()

    rows2, cols2, vals2, b_ub = [], [], [], []
    for t in range(N):                                # G + D - C >= (L-PV)·Δt
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
    G = res.x[offG:offG + N].reshape(ndays, N_SLOT)
    C = res.x[offC:offC + N].reshape(ndays, N_SLOT)
    D = res.x[offD:offD + N].reshape(ndays, N_SLOT)
    E = res.x[offE:offE + N + 1]
    p = price_mat[T0_DAY:]
    energy = float(np.sum(G[T0_DAY:] * p))
    deg = c_deg * float(D[T0_DAY:].sum())
    return dict(G=G, C=C, D=D, E=E, energy_cost=energy, deg_cost=deg,
                total=energy + deg, throughput=float(D[T0_DAY:].sum()),
                throughput_full=float(D.sum()))


def main():
    cdeg_list = [float(x) for x in sys.argv[1:]] or [0.0, 0.1, 0.2, 0.3, 0.5]
    price_all = np.tile(load_fj1()["price"], (365, 1))
    DATES, LOAD_all, PV_all = load_fj2()
    pv0_all = build_pv_forecast_stage()[0]
    # 与 §5.2/§5.4 同口径：LP 建在**全年 365 天**（首末 6000 kWh），报告统计 334 天
    price_mat, LOAD, PV, pv0 = price_all, LOAD_all, PV_all, pv0_all

    out = ["===== 储能循环寿命（老化）成本对调度的影响 =====",
           "目标：min Σp·G + c_deg·Σ D（度电成本模型，c_deg = 每 kWh 放电的老化成本）",
           "口径：全年 365 天连续储能、首末 6000 kWh；统计 2025-02-01~12-31（334 天）", ""]

    rows_det, rows_fc = [], []
    for c_deg in cdeg_list:
        r_det = _global_lp_deg(price_mat, LOAD, PV, c_deg)          # 完美预见
        r_fc = _global_lp_deg(price_mat, LOAD, pv0, c_deg)          # 点预报计划
        rows_det.append((c_deg, r_det))
        rows_fc.append((c_deg, r_fc))
        out += ["---- c_deg = %.2f 元/kWh ----" % c_deg,
                "  [完美预见] 购电费 %.2f 万元 | 储能吞吐 %.1f 万 kWh | 老化成本 %.2f 万元 | 总成本 %.2f 万元"
                % (r_det['energy_cost'] / 1e4, r_det['throughput'] / 1e4,
                   r_det['deg_cost'] / 1e4, r_det['total'] / 1e4),
                "  [点预报]   购电费 %.2f 万元 | 储能吞吐 %.1f 万 kWh | 老化成本 %.2f 万元 | 总成本 %.2f 万元"
                % (r_fc['energy_cost'] / 1e4, r_fc['throughput'] / 1e4,
                   r_fc['deg_cost'] / 1e4, r_fc['total'] / 1e4), ""]

    base = rows_det[0][1]
    out += ["---- 相对 c_deg = 0 的变化（完美预见）----"]
    for c_deg, r in rows_det:
        out.append("  c_deg=%.2f: 吞吐 %+.1f%%，购电费 %+.2f 万元，总成本（含老化）%+.2f 万元"
                   % (c_deg,
                      100.0 * (r['throughput'] - base['throughput']) / max(base['throughput'], 1e-9),
                      (r['energy_cost'] - base['energy_cost']) / 1e4,
                      (r['total'] - base['total']) / 1e4))
    open("_degradation_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图：吞吐量与总成本随 c_deg 的变化 ----
    x = [r[0] for r in rows_det]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(x, [r[1]['throughput'] / 1e4 for r in rows_det], "o-", label="完美预见")
    ax[0].plot(x, [r[1]['throughput'] / 1e4 for r in rows_fc], "s--", label="点预报")
    ax[0].set_xlabel("老化成本 $c_{deg}$（元/kWh）"); ax[0].set_ylabel("储能年吞吐量（万 kWh）")
    ax[0].set_title("老化成本抑制储能吞吐"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(x, [r[1]['energy_cost'] / 1e4 for r in rows_det], "o-", label="购电费")
    ax[1].plot(x, [r[1]['deg_cost'] / 1e4 for r in rows_det], "^-", label="老化成本")
    ax[1].plot(x, [r[1]['total'] / 1e4 for r in rows_det], "s-", label="总成本（含老化）")
    ax[1].set_xlabel("老化成本 $c_{deg}$（元/kWh）"); ax[1].set_ylabel("万元")
    ax[1].set_title("购电费与老化成本的此消彼长"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_储能老化成本." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_储能老化成本.png")


if __name__ == "__main__":
    main()
