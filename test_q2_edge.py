# -*- coding: utf-8 -*-
"""问题2 模型（solve_q2.solve_day）的边界/鲁棒性测试，用于论文的模型验证与敏感性分析。

三个测试（均用附件1 的单日电价/负载/光伏，储能首末回到初值）：
  测试A：光伏全天为 0 —— 验证仅靠外网购电即可满足负载并维持储能首末平衡。
  测试B：负载全天为 0 —— 验证模型不会"疯狂购电给储能充电"（应近似 0 购电、光伏全弃）。
  测试C：储能容量极小(1200 kWh) —— 验证储能接近无用时长模型仍能正常求解。

输出：
  _q2_edge_test.txt   —— 文字校验报告(UTF-8)。
  figures/fig_testA_光伏为零.{png,pdf}  等三张图。
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, DT, N_SLOT, E_INIT, E_MAX, E_MIN
import solve_q2

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def hours(n=N_SLOT):
    return (np.arange(n) + 0.5) * 10 / 60.0


def verify(name, r, load, pv, E0, E1, E_min, E_max, expect_g="?", expect_note=""):
    """校验一次测试结果，返回文字报告与 PASS/FAIL 摘要。"""
    G, C, D, E = r['G'], r['C'], r['D'], r['E']
    slack = G + pv * DT + D - load * DT - C          # 平衡盈余(≥0 为满足)
    bal_ok = bool((slack >= -1e-6).all())
    soc_ok = bool((E >= E_min - 1e-6).all() and (E <= E_max + 1e-6).all())
    end_ok = bool(abs(E[0] - E0) < 1e-6 and abs(E[-1] - E1) < 1e-6)
    curtail = float(slack[slack > 1e-9].sum())        # 弃电(盈余)量

    lines = ["===== %s =====" % name]
    lines.append("求解状态      : 成功 (scipy.linprog/HiGHS)")
    lines.append("功率平衡(≥)   : %s  最小盈余=%.2e kWh" % ("满足" if bal_ok else "违反!", slack.min()))
    lines.append("储能首末      : E[0]=%.2f  E[144]=%.2f  (要求 %.1f / %.1f) -> %s"
                 % (E[0], E[-1], E0, E1, "满足" if end_ok else "违反!"))
    lines.append("SOC 范围      : [%.2f, %.2f]  (限 [%.1f, %.1f]) -> %s"
                 % (E.min(), E.max(), E_min, E_max, "满足" if soc_ok else "违反!"))
    lines.append("全天购电量    : %.2f kWh  购电费 %.2f 元" % (G.sum(), r['cost']))
    lines.append("储能充/放电量 : 充 %.2f kWh  放 %.2f kWh" % (C.sum(), D.sum()))
    lines.append("弃电(盈余)量  : %.2f kWh" % curtail)
    lines.append("光伏总量      : %.2f kWh  负载总量 %.2f kWh" % (pv.sum() * DT, load.sum() * DT))
    if expect_g == "zero":
        g_ok = float(G.sum()) < 1e-6
        lines.append("关键判断      : 购电量应≈0 -> %s" % ("通过(未疯狂购电)" if g_ok else "异常(存在购电)"))
    if expect_g == "positive":
        g_ok = float(G.sum()) > 1e-6
        lines.append("关键判断      : 购电量应>0(购电满足负载) -> %s" % ("通过" if g_ok else "异常"))
    if expect_note:
        lines.append("说明          : %s" % expect_note)
    return "\n".join(lines)


def plot_test(fname, title, r, load, pv, E0, E_min, E_max):
    """绘制单个测试的两面板图：(a) 购电/充放电/负载/光伏；(b) SOC 轨迹。"""
    G, C, D, E = r['G'], r['C'], r['D'], r['E']
    h = hours()
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # (a) 购电 + 储能充放电 + 负载/光伏背景
    ax[0].step(h, pv, where="mid", label="光伏 PV", color="#e67e22", lw=1.0, ls="--")
    ax[0].step(h, load, where="mid", label="负载 L", color="#2c3e50", lw=1.1, ls=":")
    ax[0].step(h, G / DT, where="mid", label="购电 G", color="#2980b9", lw=1.6)
    ax[0].bar(h, D / DT, width=1 / 6 * 0.9, label="放电 D", color="#27ae60")
    ax[0].bar(h, -C / DT, width=1 / 6 * 0.9, label="充电 −C", color="#8e44ad")
    ax[0].set_ylabel("功率 (kW)")
    ax[0].set_title("%s\n(a) 购电与储能充放电（负载/光伏为背景）" % title)
    ax[0].legend(fontsize=8, ncol=3, loc="upper right")
    ax[0].grid(alpha=0.3)

    # (b) SOC 轨迹
    ax[1].plot(h, E[:-1], color="#16a085", lw=1.8, label="储电量 E")
    ax[1].axhline(E0, color="gray", ls="--", lw=0.9, label="初始/终值")
    ax[1].axhline(E_min, color="r", ls=":", lw=0.9)
    ax[1].axhline(E_max, color="r", ls=":", lw=0.9)
    ax[1].set_ylabel("储电量 (kWh)")
    ax[1].set_xlabel("时刻 (h)")
    ax[1].set_title("(b) 储能 SOC 轨迹")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for a in ax:
        a.set_xlim(0, 24)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / (fname + "." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已生成", fname)


def main():
    d1 = load_fj1()
    price, load, pv = d1['price'], d1['load'], d1['pv']
    out = []

    # ---- 测试A：光伏全天为 0 ----
    pv0 = np.zeros_like(pv)
    rA = solve_q2.solve_day(price, load, pv0)
    out.append(verify("测试A：光伏全天为 0（仅靠购电满足负载）", rA, load, pv0,
                      E_INIT, E_INIT, E_MIN, E_MAX, expect_g="positive"))
    plot_test("fig_testA_光伏为零", "测试A：光伏全天为 0（仅靠购电满足负载）",
              rA, load, pv0, E_INIT, E_MIN, E_MAX)

    # ---- 测试B：负载全天为 0 ----
    load0 = np.zeros_like(load)
    rB = solve_q2.solve_day(price, load0, pv)
    out.append(verify("测试B：负载全天为 0（是否疯狂购电充电）", rB, load0, pv,
                      E_INIT, E_INIT, E_MIN, E_MAX, expect_g="zero",
                      expect_note="购电=0；储能少量充放电为退化解（零成本回收光伏，调度非唯一），关键结论是模型不会购电给储能充电。"))
    plot_test("fig_testB_负载为零", "测试B：负载全天为 0（验证不疯狂购电充电）",
              rB, load0, pv, E_INIT, E_MIN, E_MAX)

    # ---- 测试C：储能容量极小 1200 kWh ----
    E_max_s, E_min_s, E_init_s = 1200.0, 0.0, 600.0
    rC = solve_q2.solve_day(price, load, pv, E0=E_init_s, E1=E_init_s,
                            E_min=E_min_s, E_max=E_max_s)
    out.append(verify("测试C：储能容量极小(1200 kWh，SOC∈[0,1200])", rC, load, pv,
                      E_init_s, E_init_s, E_min_s, E_max_s, expect_g="positive"))
    plot_test("fig_testC_储能极小", "测试C：储能容量极小（1200 kWh）",
              rC, load, pv, E_init_s, E_min_s, E_max_s)

    open("_q2_edge_test.txt", "w", encoding="utf-8").write("\n\n".join(out))
    print("报告已写出 _q2_edge_test.txt")


if __name__ == "__main__":
    main()
