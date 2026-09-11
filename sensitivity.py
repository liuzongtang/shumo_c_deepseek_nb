# -*- coding: utf-8 -*-
"""灵敏度分析：储能容量 / 储能效率 / 紧急购电价倍数 对问题二鲁棒方案成本的影响。

- 容量、效率需重新求解全年连续 LP（计划 + 重调度），故 monkeypatch solve_q4 的全局
  常量 E_MAX/E_MIN/ETA 后调用 run_robust。
- 紧急购电价倍数是费用结算中的常数因子，不改变最优调度（重调度目标为 min Σ p·e），
  故只需按比例缩放紧急费，无需重求解。

产出：_sensitivity_summary.txt + figures/fig_sensitivity.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_fj1, load_fj2
import solve_q4
from common import build_pv_forecast_stage

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 附件1 逐日同价（问题二电价环境）
d1 = load_fj1()
price_mat = np.tile(d1["price"], (365, 1))
dates, load, pv = load_fj2()
solve_q4.PRICE_MAT = price_mat
solve_q4.LOAD = load
solve_q4.PV = pv
solve_q4.DATES = dates

pvf = build_pv_forecast_stage()
pv0 = pvf[0]
p = price_mat[31:]                      # 2.1–12.31（334 天）

E_MAX0, E_MIN0, ETA0, E_INIT0 = 10800.0, 1200.0, 0.9, 6000.0


def robust_cost(eta, e_max, e_min, e_init=E_INIT0):
    solve_q4.ETA = eta
    solve_q4.E_MAX = e_max
    solve_q4.E_MIN = e_min
    solve_q4.E_INIT = e_init
    rob = solve_q4.run_robust(pv0)
    plan = float(np.sum(rob["G"] * p))
    em_kwh = float(rob["e"].sum())
    em_fee_1x = float(np.sum(rob["e"] * p))
    return plan, em_kwh, em_fee_1x


lines = []
def log(s=""):
    lines.append(s)
    print(s)


log("===== 灵敏度分析（问题二鲁棒 · 附件1 逐日同价）=====")
log("")

# ---- 1. 储能容量 ----
log("【1】储能容量敏感性（效率=0.9，倍数=5）")
log("  容量缩放 | E_MAX | E_MIN | 计划费(元) | 紧急电量(kWh) | 紧急费5x(元) | 总费(元)")
cap_rows = []
for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
    plan, em_kwh, em_1x = robust_cost(ETA0, E_MAX0 * scale, E_MIN0 * scale, E_INIT0 * scale)
    em5 = 5 * em_1x
    tot = plan + em5
    cap_rows.append((scale, plan, em_kwh, em5, tot))
    log("  %5.2f   | %5.0f | %4.0f | %12.2f | %12.2f | %12.2f | %12.2f"
        % (scale, E_MAX0 * scale, E_MIN0 * scale, plan, em_kwh, em5, tot))

# ---- 2. 储能效率 ----
log("")
log("【2】储能效率敏感性（容量=100%，倍数=5）")
log("  效率 η | 计划费(元) | 紧急电量(kWh) | 紧急费5x(元) | 总费(元)")
eff_rows = []
for eta in [0.80, 0.85, 0.90, 0.95]:
    plan, em_kwh, em_1x = robust_cost(eta, E_MAX0, E_MIN0)
    em5 = 5 * em_1x
    tot = plan + em5
    eff_rows.append((eta, plan, em_kwh, em5, tot))
    log("  %.2f | %12.2f | %12.2f | %12.2f | %12.2f"
        % (eta, plan, em_kwh, em5, tot))

# ---- 3. 紧急购电价倍数 ----
log("")
log("【3】紧急购电价倍数敏感性（容量=100%，效率=0.9；无需重求解）")
plan0, em_kwh0, em_1x0 = robust_cost(ETA0, E_MAX0, E_MIN0)
log("  计划费 = %.2f 元；紧急费(1倍基准) = %.2f 元；紧急电量 = %.2f kWh"
    % (plan0, em_1x0, em_kwh0))
log("  倍数 | 紧急费(元) | 总费(元)")
mul_rows = []
for m in [3, 4, 5, 6, 7]:
    em = m * em_1x0
    tot = plan0 + em
    mul_rows.append((m, em, tot))
    log("  %d | %12.2f | %12.2f" % (m, em, tot))

open("_sensitivity_summary.txt", "w", encoding="utf-8").write("\n".join(lines))

# ---- 图 ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
sc = [r[0] for r in cap_rows]
tot = [r[4] / 1e4 for r in cap_rows]
ax.plot(sc, tot, "o-", color="#d62728", lw=2)
ax.set_xlabel("储能容量缩放（1.0 = 12 MWh 名义容量）")
ax.set_ylabel("总购电费（万元）")
ax.set_title("(a) 储能容量")
ax.grid(alpha=0.3)

ax = axes[1]
et = [r[0] for r in eff_rows]
tot = [r[4] / 1e4 for r in eff_rows]
ax.plot(et, tot, "o-", color="#1f77b4", lw=2)
ax.set_xlabel("充放电效率 η")
ax.set_ylabel("总购电费（万元）")
ax.set_title("(b) 储能效率")
ax.grid(alpha=0.3)

ax = axes[2]
mm = [r[0] for r in mul_rows]
tot = [r[2] / 1e4 for r in mul_rows]
ax.plot(mm, tot, "o-", color="#2ca02c", lw=2)
ax.set_xlabel("紧急购电价倍数")
ax.set_ylabel("总购电费（万元）")
ax.set_title("(c) 紧急购电价倍数")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig_sensitivity.png", dpi=150, bbox_inches="tight")
print("\n图已保存 figures/fig_sensitivity.png")
print("done")
