# -*- coding: utf-8 -*-
"""生成论文用图（中文标签、高清 PNG + PDF）。
图1 问题1单日调度全景；图2 问题2全年购电费构成；图3 问题3滚动调整(3.20)；
图4 问题4 SOC跨日轨迹；图5 问题4电价波动特征；图6 光伏预报误差。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, load_fj2, load_fj4, DT, N_SLOT, E_INIT
import solve_q1, solve_q2, solve_q3, solve_q4

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / (name + "." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已生成", name)


def hours(n=N_SLOT):
    """槽中心时刻(小时)。"""
    return (np.arange(n) + 0.5) * 10 / 60.0


# ============ 数据准备 ============
d1 = load_fj1()
price1, load1, pv1 = d1["price"], d1["load"], d1["pv"]
r1 = solve_q1.solve_day(price1, load1, pv1)

DATES, LOAD, PV = load_fj2()
_, PRICE4 = load_fj4()

# 问题2/3 全局（全年连续储能，附件1 每日同价）—— 修正版，与 solve_q23_continuous.py 一致
from solve_q3 import build_pv_forecast_stage
price_mat1 = np.tile(price1, (365, 1))
solve_q4.PRICE_MAT = price_mat1
solve_q4.DATES = DATES
solve_q4.LOAD = LOAD
solve_q4.PV = PV
pvf = build_pv_forecast_stage()
pv0 = pvf[0]                                        # 0:00 光伏预报 (365,144)
rob2 = solve_q4.run_robust(pv0)                     # 连续储能鲁棒
res3 = solve_q4.run_rolling(pvf)                    # 连续储能滚动

# 问题4 全局(用于跨日SOC，附件4 波动电价)
solve_q4.PRICE_MAT = PRICE4
solve_q4.DATES = DATES
solve_q4.LOAD = LOAD
solve_q4.PV = PV
rob4 = solve_q4.run_robust(pv0)          # 鲁棒：0:00预报计划 + 实际光伏全局重调度

# ============ 图1 问题1 单日调度全景 ============
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
h = hours()
# (a) 电价
ax[0, 0].step(h, price1, where="mid", color="#c0392b", lw=1.4)
ax[0, 0].set_ylabel("电价 (元/kWh)")
ax[0, 0].set_title("(a) 分时电价")
ax[0, 0].grid(alpha=0.3)
# (b) 负载 vs 光伏
ax[0, 1].plot(h, load1, label="小区负载", color="#2c3e50", lw=1.2)
ax[0, 1].plot(h, pv1, label="光伏预测", color="#e67e22", lw=1.2)
ax[0, 1].set_ylabel("功率 (kW)")
ax[0, 1].set_title("(b) 负载与光伏")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=0.3)
# (c) 购电 + 充放电
ax[1, 0].bar(h, r1["G"] / DT, width=1 / 6 * 0.9, label="计划购电 G", color="#2980b9")
ax[1, 0].bar(h, r1["D"] / DT, width=1 / 6 * 0.9, label="储能放电 D", color="#27ae60")
ax[1, 0].bar(h, -r1["C"] / DT, width=1 / 6 * 0.9, label="储能充电 −C", color="#8e44ad")
ax[1, 0].set_ylabel("功率 (kW)")
ax[1, 0].set_title("(c) 购电与储能充放电")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=0.3)
# (d) SOC
ax[1, 1].plot(h, r1["E"][:-1], color="#16a085", lw=1.6)
ax[1, 1].axhline(6000, color="gray", ls="--", lw=0.8, label="初始/终值 6000")
ax[1, 1].axhline(1200, color="r", ls=":", lw=0.8)
ax[1, 1].axhline(10800, color="r", ls=":", lw=0.8)
ax[1, 1].set_ylabel("储电量 (kWh)")
ax[1, 1].set_title("(d) 储能 SOC")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.3)
for a in ax.ravel():
    a.set_xlim(0, 24)
    a.set_xlabel("时刻 (h)")
fig.suptitle("问题1 单日计划购电与储能调度", y=1.02, fontsize=12)
fig.tight_layout()
save(fig, "fig1_问题1_单日调度")

# ============ 图2 问题2 全年购电费构成(月度) ============
dates = DATES[31:]
months = [getattr(d, "month", None) or dt.datetime.strptime(str(d)[:10], "%Y-%m-%d").month
          for d in dates]
months = np.array(months)
G2, e2 = rob2["G"], rob2["e"]
plan_daily = np.sum(G2 * price1[None, :], axis=1)
em_daily = np.sum(e2 * price1[None, :], axis=1) * 5.0
m_labels = ["%d月" % m for m in range(2, 13)]
plan_m = [plan_daily[months == m].sum() / 1e4 for m in range(2, 13)]
em_m = [em_daily[months == m].sum() / 1e4 for m in range(2, 13)]
x = np.arange(len(m_labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x, plan_m, label="计划购电费", color="#2980b9")
ax.bar(x, em_m, bottom=plan_m, label="紧急购电费(5倍)", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(m_labels)
ax.set_ylabel("购电费 (万元)")
ax.set_title("问题2 全年购电费构成（鲁棒方案，逐月）")
ax.legend()
ax.grid(alpha=0.3, axis="y")
save(fig, "fig2_问题2_全年购电费构成")

# ============ 图3 问题3 滚动调整(3.20) ============
idx320 = [i for i, d in enumerate(DATES) if str(d).replace("-", ".").startswith("2025.3.20")
          or str(d).startswith("2025-03-20")][0]
k320 = idx320 - 31
Gp = res3["G_plan"][k320]
Ga = res3["G_adj"][k320]
e3 = res3["e"][k320]
up = res3["up"][k320]
fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
ax[0].plot(h, Gp / DT, label="计划购电 G_plan(0:00预报)", color="#2980b9", lw=1.2)
ax[0].plot(h, Ga / DT, label="调整购电 G_adj(6/12/18点滚动)", color="#e67e22", lw=1.2)
ax[0].set_ylabel("购电功率 (kW)")
ax[0].set_title("问题3  2025-03-20 计划与调整购电")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[1].bar(h, up / DT, width=1 / 6 * 0.9, label="上调量 up(1.5倍)", color="#e67e22")
ax[1].bar(h, e3 / DT, width=1 / 6 * 0.9, label="紧急购电 e(5倍)", color="#e74c3c")
ax[1].set_ylabel("功率 (kW)")
ax[1].set_xlabel("时刻 (h)")
ax[1].set_title("上调量与紧急购电")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
for a in ax:
    a.set_xlim(0, 24)
fig.tight_layout()
save(fig, "fig3_问题3_滚动调整_0320")

# ============ 图4 问题4 SOC跨日轨迹(连续储能) ============
# 取 2.1-2.14 两周
nd = 14
Efull = rob4["E"]  # 鲁棒方案实际执行SOC (334*144+1,)
fig, ax = plt.subplots(figsize=(11, 5.5))
tt = np.arange(nd * N_SLOT + 1) * 10 / 60.0
ax.plot(tt, Efull[:nd * N_SLOT + 1], color="#16a085", lw=1.4)
# 每日边界虚线
for d in range(nd + 1):
    ax.axvline(d * 24, color="gray", ls="--", lw=0.5, alpha=0.6)
ax.axhline(6000, color="gray", ls=":", lw=0.8)
ax.set_ylabel("储电量 (kWh)")
ax.set_xlabel("时间 (h，自 2025-02-01 0:00 起)")
ax.set_title("问题4 连续储能 SOC 跨日轨迹（鲁棒方案·实际执行，2.1–2.14）")
ax.set_ylim(0, 12000)
ax.grid(alpha=0.3)
# 次轴画日平均电价(阶梯，与SOC日变化对应)
ax2 = ax.twinx()
dmean_win = PRICE4[31:31 + nd].mean(axis=1)
x_step = np.arange(nd + 1) * 24.0
y_step = np.concatenate([dmean_win, dmean_win[-1:]])
ax2.step(x_step, y_step, where="post", color="#c0392b", lw=1.6, label="日平均电价")
ax2.set_ylabel("日平均电价 (元/kWh)", color="#c0392b")
ax2.tick_params(axis="y", labelcolor="#c0392b")
ax2.legend(loc="upper right", fontsize=8)
save(fig, "fig4_问题4_SOC跨日轨迹")

# ============ 图5 问题4 电价波动特征 ============
dmean4 = PRICE4.mean(axis=1)
dmin4 = PRICE4.min(axis=1)
dmax4 = PRICE4.max(axis=1)
dd = np.arange(365)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(dd, dmean4, color="#2c3e50", lw=1.0, label="日平均电价")
ax.fill_between(dd, dmin4, dmax4, color="#2980b9", alpha=0.25, label="日内[min,max]区间")
ax.set_xlabel("日期序号 (2025-01-01 起)")
ax.set_ylabel("电价 (元/kWh)")
ax.set_title("问题4 附件4 全年波动电价")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig5_问题4_电价波动")

# ============ 图6 光伏预报误差 ============
err = pv0[31:] - PV[31:]
fig, ax = plt.subplots(2, 1, figsize=(10.5, 6))
ax[0].hist(err.ravel(), bins=60, color="#8e44ad", alpha=0.8)
ax[0].set_xlabel("预报误差 (kW)")
ax[0].set_ylabel("频数")
ax[0].set_title("问题2/3 光伏0:00预报误差分布（预报−实际）")
ax[0].grid(alpha=0.3)
# 逐日平均绝对误差
dae = np.abs(err).mean(axis=1)
ax[1].plot(np.arange(len(dae)), dae, color="#e67e22", lw=0.8)
ax[1].set_xlabel("日期序号 (2.1 起)")
ax[1].set_ylabel("日平均绝对误差 (kW)")
ax[1].set_title("逐日光伏预报平均绝对误差")
ax[1].grid(alpha=0.3)
fig.tight_layout()
save(fig, "fig6_光伏预报误差")

print("全部图生成完毕 -> figures/")
