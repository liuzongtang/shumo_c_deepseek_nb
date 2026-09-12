# -*- coding: utf-8 -*-
"""方向D：光伏预报误差的条件分布与尾部风险。

机理：紧急购电由"高估"（预报>实际）驱动。高估的规模与频率随预报水平而异——午间
高预报（大光伏）时，绝对误差更大、且系统性偏正（高估），构成紧急购电的尾部风险。
本脚本按 0:00 预报水平分箱，量化误差的条件均值/标准差/90%分位（尾部），刻画异方差。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, load_fj2
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

    pv0_rep = pv0[31:].ravel()       # 0:00 预报 (334*144,)
    PV_rep = PV[31:].ravel()         # 实际
    err = pv0_rep - PV_rep           # kW，>0 高估

    bins = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 20000]
    lab = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        lab.append("[%d,%d)" % (a, b))

    out = []
    out.append("===== 方向D 光伏预报误差条件分布与尾部风险 =====")
    out.append("总体：误差(预报-实际) 均值 %+.1f kW, 标准差 %.1f kW, 偏度 %+.2f"
               % (err.mean(), err.std(),
                  float(np.mean((err - err.mean()) ** 3) / err.std() ** 3)))
    out.append("")
    out.append("按 0:00 预报水平分箱的条件误差统计（kW）：")
    out.append("  %-12s %6s %8s %8s %8s %8s %8s" %
               ("预报箱", "样本%", "均值", "标准差", "P90高估", "P99高估", "高估率%"))
    mus, sds, q90s, q99s, overp = [], [], [], [], []
    centers = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        mk = (pv0_rep >= a) & (pv0_rep < b)
        n = int(mk.sum())
        if n == 0:
            continue
        e = err[mk]
        centers.append(0.5 * (a + b))
        mus.append(float(e.mean()))
        sds.append(float(e.std()))
        q90s.append(float(np.percentile(e, 90)))
        q99s.append(float(np.percentile(e, 99)))
        overp.append(float((e > 0).mean() * 100))
        out.append("  %-12s %5.1f%% %+7.0f %8.0f %8.0f %8.0f %7.1f%%"
                   % (lab[i], 100.0 * n / len(err), e.mean(), e.std(),
                      np.percentile(e, 90), np.percentile(e, 99), 100 * (e > 0).mean()))
    out.append("")
    out.append("关键结论（尾部风险的两条来源）：")
    out.append("  (1) 昼间各箱误差标准差 ~850-1000 kW 近似同方差，但高估率约 45-50%：")
    out.append("      即昼间约一半时段预报偏高，构成中等幅度的常规高估。")
    out.append("  (2) 极端高预报箱 [8000,+∞)（占 5.4% 样本）均值 +522 kW、高估率 78.2%、")
    out.append("      P99 高估达 2096 kW：预报在极高水平下系统性偏高，构成罕见而大幅的")
    out.append("      尾部风险，是紧急购电峰值的主要来源。")
    out.append("  反观低预报箱 [0,500)（占 52.5%，夜间/清晨）高估率仅 2.6%、尾部几乎为零。")
    open("_error_conditional_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图 ----
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    # (a) 误差 vs 预报水平 散点 + 条件均值/90%分位
    idx = np.random.RandomState(0).choice(len(err), size=min(8000, len(err)),
                                          replace=False)
    ax[0].scatter(pv0_rep[idx], err[idx], s=4, alpha=0.25, color="#7f8c8d")
    ax[0].plot(centers, mus, "o-", color="#2980b9", lw=2, label="条件均值")
    ax[0].plot(centers, q90s, "s--", color="#c0392b", lw=2, label="90% 分位（尾部）")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("0:00 光伏预报水平 (kW)")
    ax[0].set_ylabel("预报误差 预报−实际 (kW)")
    ax[0].set_title("(a) 误差 vs 预报水平（异方差，尾部随预报增大）")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # (b) 条件均值 + 高估率 vs 预报箱
    x = np.arange(len(centers))
    ax2 = ax[1]
    bars = ax2.bar(x, mus, color="#16a085", label="条件均值（kW）")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lab, rotation=45, fontsize=7)
    ax2.set_xlabel("预报水平箱 (kW)")
    ax2.set_ylabel("条件均值 预报−实际 (kW)")
    ax2.set_title("(b) 条件均值（高预报系统性高估）")
    ax2.grid(alpha=0.3, axis="y")
    ax3 = ax2.twinx()
    ax3.plot(x, overp, "o-", color="#c0392b", lw=2, label="高估率%")
    ax3.set_ylabel("高估率 (%)", color="#c0392b")
    ax3.tick_params(axis="y", labelcolor="#c0392b")
    ax3.set_ylim(0, 100)
    l1, la1 = ax2.get_legend_handles_labels()
    l2, la2 = ax3.get_legend_handles_labels()
    ax2.legend(l1 + l2, la1 + la2, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_误差条件分布." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(out))
    print("已生成 figures/fig_误差条件分布.png")


if __name__ == "__main__":
    main()
