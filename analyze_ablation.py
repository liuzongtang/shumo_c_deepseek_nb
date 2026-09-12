# -*- coding: utf-8 -*-
"""方向1：消融 / 贡献分解——量化 储能、光伏预报、午间滚动预报 各自的经济价值。

五个配置（附件1 逐日同价，连续储能，334 天报告期 2.1–12.31）：
  (1) 无储能 · 完美预见          —— 物理下界（无储能、已知实际光伏）
  (2) 有储能 · 完美预见          —— 确定性下界（run_deterministic）
  (3) 无储能 · 仅0:00预报         —— 无储能时预报误差的代价
  (4) 有储能 · 仅0:00预报（鲁棒）  —— 问题2 当前结果
  (5) 有储能 · 0/6/12/18滚动       —— 问题3 当前结果

贡献（相邻配置之差）：
  储能价值(完美预见下) = (1)−(2)；储能价值(不确定下) = (3)−(4)
  预报误差代价          = (4)−(2)；午间滚动预报价值 = (4)−(5)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, load_fj2, DT
import solve_q4
from common import build_pv_forecast_stage

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def main():
    d1 = load_fj1()
    price1 = d1["price"]
    DATES, LOAD, PV = load_fj2()
    price_mat = np.tile(price1, (365, 1))
    solve_q4.PRICE_MAT = price_mat
    solve_q4.LOAD = LOAD
    solve_q4.PV = PV
    solve_q4.DATES = DATES
    pvf = build_pv_forecast_stage()
    pv0 = pvf[0]

    L, PVa, p = LOAD, PV, price_mat
    sl = slice(31, 365)                      # 报告期 334 天

    # (1) 无储能·完美预见：逐段买负荷缺口 max(0, L-PV)
    G1 = np.maximum(0.0, (L - PVa) * DT)
    c1 = float(np.sum(G1[sl] * p[sl]))

    # (2) 有储能·完美预见（确定性）
    det = solve_q4.run_deterministic()
    c2 = float(det["cost"])

    # (3) 无储能·仅0:00预报：计划固定 + 5倍紧急（高估光伏→缺额）
    G3p = np.maximum(0.0, (L - pv0) * DT)
    e3 = np.maximum(0.0, (pv0 - PVa) * DT)
    c3 = float(np.sum(G3p[sl] * p[sl]) + 5.0 * np.sum(e3[sl] * p[sl]))

    # (4) 有储能·仅0:00预报（鲁棒）
    rob = solve_q4.run_robust(pv0)
    c4 = float(np.sum(rob["G"] * p[sl])) + 5.0 * float(np.sum(rob["e"] * p[sl]))

    # (5) 有储能·滚动
    roll = solve_q4.run_rolling(pvf)
    c5 = (float(np.sum(roll["G_plan"] * p[sl]))
          + float(np.sum((1.5 * roll["up"] + 0.5 * roll["dn"]) * p[sl]))
          + 5.0 * float(np.sum(roll["e"] * p[sl])))

    # 贡献
    val_storage_pf = c1 - c2        # 储能价值（完美预见下）
    val_storage_uc = c3 - c4        # 储能价值（不确定性下）
    cost_forecast = c4 - c2         # 预报误差代价
    val_rolling = c4 - c5           # 午间滚动预报价值

    names = ["无储能\n完美预见", "有储能\n完美预见", "无储能\n仅0:00预报",
             "有储能\n仅0:00预报", "有储能\n滚动0/6/12/18"]
    costs = [c1, c2, c3, c4, c5]
    w = [c / 1e4 for c in costs]

    out = []
    out.append("===== 方向1 消融/贡献分解（附件1 逐日同价，334 天）=====")
    for n, c in zip(names, costs):
        out.append("  %-18s %12.2f 元 (%.2f 万元)" % (n.replace("\n", ""), c, c / 1e4))
    out.append("")
    out.append("贡献分解：")
    out.append("  储能价值（完美预见下）  = (1)-(2) = %.2f 元 (%.2f 万元)"
               % (val_storage_pf, val_storage_pf / 1e4))
    out.append("  储能价值（不确定性下）  = (3)-(4) = %.2f 元 (%.2f 万元)"
               % (val_storage_uc, val_storage_uc / 1e4))
    out.append("  光伏预报误差代价        = (4)-(2) = %.2f 元 (%.2f 万元)"
               % (cost_forecast, cost_forecast / 1e4))
    out.append("  午间滚动预报价值        = (4)-(5) = %.2f 元 (%.2f 万元)"
               % (val_rolling, val_rolling / 1e4))
    open("_ablation_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图：水平条形 + 贡献注释 ----
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#95a5a6", "#16a085", "#95a5a6", "#2980b9", "#e67e22"]
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, w, color=colors, height=0.62)
    for yi, wi in zip(y, w):
        ax.text(wi + 8, yi, "%.2f 万" % wi, va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("全年总购电费（万元，2025-02-01~12-31）")
    ax.set_title("消融分析：储能、光伏预报与滚动调整的贡献")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, max(w) * 1.13)
    # 贡献注释（右侧/底部）
    ann = ("储能价值(完美预见) %.1f 万\n储能价值(不确定下) %.1f 万\n"
           "预报误差代价 %.1f 万\n午间滚动价值 %.1f 万")
    ax.text(0.99, 0.03, ann % (val_storage_pf / 1e4, val_storage_uc / 1e4,
                               cost_forecast / 1e4, val_rolling / 1e4),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf6e3", ec="#e0c060", alpha=0.95))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_消融贡献分解." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_消融贡献分解.png")


if __name__ == "__main__":
    main()
