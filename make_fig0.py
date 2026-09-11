# -*- coding: utf-8 -*-
"""生成 fig0_思路图：展示四问递进关系与统一 LP 框架。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

# 配色
C_ROOT = "#2c3e50"
C_Q = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad"]
C_FOOT = "#16a085"
C_EDGE = "#7f8c8d"


def box(ax, x, y, w, h, text, fc, tc="white", fs=12, lw=1.4, bold=False):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.06,rounding_size=0.12",
                       linewidth=lw, edgecolor="white", facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.6)


def arrow(ax, x1, y1, x2, y2, color=C_EDGE):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=16, linewidth=1.6, color=color, zorder=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # ---- 顶部标题 ----
    ax.text(6.25, 6.85, "微网与外部电网电力调控策略 · 统一建模思路",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1a252f")

    # ---- 根节点：统一 LP 框架 ----
    box(ax, 4.05, 5.45, 4.4, 0.95,
        "统一线性规划（LP）框架\n目标 min Σ pₜ·Gₜ（购电费最小）\n约束：功率平衡 + 储能 SOC 动态",
        C_ROOT, fs=11.5, bold=True)

    # ---- 四个问题节点（2×2）----
    qw, qh = 3.7, 1.05
    qx = [0.9, 5.35, 0.9, 5.35]          # 列1, 列2
    qy = [3.55, 3.55, 1.95, 1.95]        # 上排, 下排
    qtitle = ["问题 1 · 单日计划购电", "问题 2 · 全年计划 + 紧急购电",
              "问题 3 · 滚动调整购电", "问题 4 · 波动电价购电"]
    qsub = ["静态 LP：光伏/负载/电价全知\n0:00=24:00 电量，谷充峰放",
            "两阶段鲁棒 LP：0:00 按预报定计划\n揭晓后重调度 + 5 倍紧急购电",
            "滚动 LP：0/6/12/18 时预报\n下调违约 50% / 上调溢价 1.5 倍",
            "全年连续 LP：附件 4 波动电价\n储能跨日搬移（跨日套利）"]

    for i in range(4):
        box(ax, qx[i], qy[i], qw, qh, qtitle[i] + "\n" + qsub[i], C_Q[i], fs=10.5)
        # 根 -> 问题 连线（从根底部到各问题顶部）
        if i == 0:
            arrow(ax, 4.45, 5.45, qx[0] + qw / 2, qy[0] + qh)
        elif i == 1:
            arrow(ax, 6.05, 5.45, qx[1] + qw / 2, qy[1] + qh)
        elif i == 2:
            arrow(ax, 5.2, 5.45, qx[2] + qw / 2, qy[2] + qh)
        else:
            arrow(ax, 6.6, 5.45, qx[3] + qw / 2, qy[3] + qh)

    # ---- 底部：求解与验证 ----
    box(ax, 1.55, 0.45, 4.4, 0.85,
        "求解器：scipy.optimize.linprog（HiGHS 单纯形）\n精确全局最优，全年 4.8 万时段秒级求解",
        C_FOOT, fs=10.5)
    box(ax, 6.55, 0.45, 4.4, 0.85,
        "模型验证：边界测试 A/B/C\n（光伏=0 / 负载=0 / 储能极小）全部通过",
        "#7f8c8d", fs=10.5)
    for xq in (qx[2] + qw / 2, qx[3] + qw / 2):
        arrow(ax, xq, 1.95, xq, 1.3)

    fig.tight_layout(pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig0_思路图." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已生成 fig0_思路图.png/pdf")


if __name__ == "__main__":
    main()
