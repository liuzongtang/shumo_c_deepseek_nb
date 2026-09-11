# -*- coding: utf-8 -*-
"""结果复现 + 逐时段物理不变量校验（问题1~4）。

独立重跑四个问题的求解器，验证：
  (1) 关键费用/电量与论文报告值一致（可复现）；
  (2) 物理不变量逐时段成立：SOC 动态等式、SOC∈[1200,10800]、充放电≤5000kW、
      功率平衡不等式（允许弃光）、非负性、储能首末电量 E[0]=E[365]=6000、
      问题3 下调恒为 0（计划费沉没，只上调不下调）。

只读内存、不写结果文件，输出 `_validation.txt`。
"""
import numpy as np
from data_loader import (load_fj1, load_fj2, load_fj4,
                         DT, PMAX_E, E_MAX, E_MIN, E_INIT, ETA, N_SLOT)
import solve_q1, solve_q4
from common import build_pv_forecast_stage, MERGE_EPS

TOL = 1e-6          # 数值容差（能量/kWh 量级，高于 HiGHS 求解容差）
FEE_TOL = 1.0       # 费用复现容差（元，结果量级 ~1e7）
lines = []


def ok(name, cond, detail=""):
    lines.append("%s  %s%s" % ("[PASS]" if cond else "[FAIL]", name,
                               ("  " + detail) if detail else ""))
    return cond


def check_continuous(G, C, D, E, e, LOAD_mat, PV_mat, label):
    """对 (ndays,144) 全年/334 天连续储能结果做物理不变量校验。
    G=计划(或调整)购电、C/D=充/放电、E=储电量轨迹(ndays*144+1)、e=紧急购电(均 kWh)。"""
    ndays = G.shape[0]
    N = ndays * N_SLOT
    Gf, Cf, Df, ef = G.ravel(), C.ravel(), D.ravel(), e.ravel()
    Lf = LOAD_mat.ravel() * DT
    PVf = PV_mat.ravel() * DT

    # (1) SOC 动态等式：E[t+1] = E[t] + η·C[t] − D[t]/η
    resid = E[1:] - E[:-1] - ETA * Cf + Df / ETA
    ok("%s · SOC 动态等式残差≈0" % label, np.abs(resid).max() < TOL,
       "max=%.3e" % np.abs(resid).max())

    # (2) SOC 边界
    ok("%s · SOC∈[%.0f,%.0f]" % (label, E_MIN, E_MAX),
       E.min() >= E_MIN - TOL and E.max() <= E_MAX + TOL,
       "min=%.3f max=%.3f" % (E.min(), E.max()))

    # (3) 充放电功率上限（每段 ≤5000kW×Δt=833.33 kWh）
    ok("%s · 充/放电 ≤%.2f kWh/段" % (label, PMAX_E),
       Cf.max() <= PMAX_E + TOL and Df.max() <= PMAX_E + TOL,
       "Cmax=%.3f Dmax=%.3f" % (Cf.max(), Df.max()))

    # (4) 功率平衡不等式（允许弃光）：G + PV·Δt + D + e ≥ L·Δt + C
    bal = Gf + PVf + Df + ef - Lf - Cf
    ok("%s · 功率平衡盈余≥0" % label, bal.min() >= -TOL,
       "min盈余=%.3e" % bal.min())

    # (5) 非负性
    ok("%s · 非负性" % label,
       Gf.min() >= -TOL and Cf.min() >= -TOL and Df.min() >= -TOL and ef.min() >= -TOL, "")


def main():
    d1 = load_fj1()
    price1, load1, pv1 = d1['price'], d1['load'], d1['pv']
    DATES, LOAD, PV = load_fj2()
    _, PRICE4 = load_fj4()
    price_mat = np.tile(price1, (365, 1))      # 附件1 逐日重复
    solve_q4.PRICE_MAT = price_mat
    solve_q4.LOAD = LOAD
    solve_q4.PV = PV
    solve_q4.DATES = DATES
    pvf = build_pv_forecast_stage()
    pv0 = pvf[0]

    lines.append("===== 结果复现与物理不变量校验 =====")
    lines.append("数值容差：能量 %.0e kWh，费用 %.0e 元" % (TOL, FEE_TOL))
    lines.append("")

    # ================= 问题1 =================
    lines.append("---- 问题1（单日计划购电）----")
    r1 = solve_q1.solve_day(price1, load1, pv1)
    ok("问题1 全天购电量 = 59482.6990 kWh", abs(r1['G'].sum() - 59482.6990) < 0.05,
       "=%.4f" % r1['G'].sum())
    ok("问题1 购电费 = 35126.9486 元", abs(r1['cost'] - 35126.9486) < 0.05,
       "=%.4f" % r1['cost'])
    ok("问题1 E[0]=E[144]=6000",
       abs(r1['E'][0] - E_INIT) < TOL and abs(r1['E'][-1] - E_INIT) < TOL,
       "E0=%.3f E144=%.3f" % (r1['E'][0], r1['E'][-1]))
    r1_resid = r1['E'][1:] - r1['E'][:-1] - ETA * r1['C'] + r1['D'] / ETA
    ok("问题1 SOC 动态等式残差≈0", np.abs(r1_resid).max() < TOL,
       "max=%.3e" % np.abs(r1_resid).max())
    r1_bal = r1['G'] + pv1 * DT + r1['D'] - load1 * DT - r1['C']
    ok("问题1 功率平衡盈余≥0", r1_bal.min() >= -TOL, "min=%.3e" % r1_bal.min())
    ok("问题1 SOC∈[1200,10800]", r1['E'].min() >= E_MIN - TOL and r1['E'].max() <= E_MAX + TOL,
       "min=%.2f max=%.2f" % (r1['E'].min(), r1['E'].max()))
    lines.append("")

    # ================= 问题2（附件1 逐日同价，连续储能） =================
    lines.append("---- 问题2（全年连续储能，附件1 同价）----")
    plan = solve_q4._global_lp(price_mat, LOAD, pv0)
    rc = solve_q4._global_recourse(plan['G'], price_mat, LOAD, PV)
    ok("问题2 全局 LP 储能首末电量 E[0]=E[365·144]=6000",
       abs(plan['E'][0] - E_INIT) < TOL and abs(plan['E'][-1] - E_INIT) < TOL,
       "E0=%.3f E_end=%.3f" % (plan['E'][0], plan['E'][-1]))
    ok("问题2 重调度储能首末电量 E[0]=E[365·144]=6000",
       abs(rc['E'][0] - E_INIT) < TOL and abs(rc['E'][-1] - E_INIT) < TOL,
       "E0=%.3f E_end=%.3f" % (rc['E'][0], rc['E'][-1]))
    check_continuous(plan['G'], rc['C'], rc['D'], rc['E'], rc['e'], LOAD, PV, "问题2·鲁棒(365天)")
    p31 = price_mat[31:]
    plan_fee2 = float(np.sum(plan['G'][31:] * p31))
    em_fee2 = float(np.sum(rc['e'][31:] * p31)) * 5.0
    ok("问题2 计划购电费 = 12549415.99 元", abs(plan_fee2 - 12549415.99) < FEE_TOL,
       "=%.2f" % plan_fee2)
    ok("问题2 紧急购电费 = 2175844.25 元", abs(em_fee2 - 2175844.25) < FEE_TOL,
       "=%.2f" % em_fee2)
    ok("问题2 总购电费 = 14725260.23 元", abs(plan_fee2 + em_fee2 - 14725260.23) < FEE_TOL,
       "=%.2f" % (plan_fee2 + em_fee2))
    ok("问题2 紧急购电量 = 575740.84 kWh",
       abs(rc['e'][31:].sum() - 575740.84) < 0.5, "=%.2f" % rc['e'][31:].sum())
    lines.append("")

    # ================= 问题3（附件1 逐日同价，滚动调整） =================
    lines.append("---- 问题3（滚动调整，附件1 同价）----")
    roll = solve_q4.run_rolling(pvf, E_plan=plan['E'])   # 复用问题2 的全局计划 SOC
    check_continuous(roll['G_adj'], roll['C'], roll['D'], roll['E'], roll['e'],
                     LOAD[31:], PV[31:], "问题3·滚动(334天)")
    plan_fee3 = float(np.sum(roll['G_plan'] * p31))
    adj_fee3 = float(np.sum((1.5 * roll['up'] + 0.5 * roll['dn']) * p31))
    em_fee3 = float(np.sum(roll['e'] * p31)) * 5.0
    ok("问题3 计划购电费 = 12549415.99 元", abs(plan_fee3 - 12549415.99) < FEE_TOL,
       "=%.2f" % plan_fee3)
    ok("问题3 调整相关费 = 444364.53 元", abs(adj_fee3 - 444364.53) < FEE_TOL,
       "=%.2f" % adj_fee3)
    ok("问题3 紧急购电费 = 1567947.09 元", abs(em_fee3 - 1567947.09) < FEE_TOL,
       "=%.2f" % em_fee3)
    ok("问题3 总购电费 = 14561727.61 元",
       abs(plan_fee3 + adj_fee3 + em_fee3 - 14561727.61) < FEE_TOL,
       "=%.2f" % (plan_fee3 + adj_fee3 + em_fee3))
    dn_sum = float(roll['dn'].sum())
    ok("问题3 下调恒为 0（计划费沉没，只上调不下调）", dn_sum < TOL, "Σdn=%.3e" % dn_sum)
    ok("问题3 上调与下调不同时发生", np.min(roll['up'] * roll['dn']) >= -TOL, "")
    lines.append("")

    # ================= 问题4（附件4 波动电价，连续储能） =================
    lines.append("---- 问题4（附件4 波动电价，连续储能）----")
    solve_q4.PRICE_MAT = PRICE4
    plan4 = solve_q4._global_lp(PRICE4, LOAD, pv0)
    rc4 = solve_q4._global_recourse(plan4['G'], PRICE4, LOAD, PV)
    p4 = PRICE4[31:]
    plan_fee4 = float(np.sum(plan4['G'][31:] * p4))
    em_fee4 = float(np.sum(rc4['e'][31:] * p4)) * 5.0
    ok("问题4·问题2 总购电费 = 15315132.18 元",
       abs(plan_fee4 + em_fee4 - 15315132.18) < FEE_TOL, "=%.2f" % (plan_fee4 + em_fee4))
    roll4 = solve_q4.run_rolling(pvf, E_plan=plan4['E'])
    plan_fee43 = float(np.sum(roll4['G_plan'] * p4))
    adj_fee43 = float(np.sum((1.5 * roll4['up'] + 0.5 * roll4['dn']) * p4))
    em_fee43 = float(np.sum(roll4['e'] * p4)) * 5.0
    ok("问题4·问题3 总购电费 = 15150992.45 元",
       abs(plan_fee43 + adj_fee43 + em_fee43 - 15150992.45) < FEE_TOL,
       "=%.2f" % (plan_fee43 + adj_fee43 + em_fee43))
    check_continuous(roll4['G_adj'], roll4['C'], roll4['D'], roll4['E'], roll4['e'],
                     LOAD[31:], PV[31:], "问题4·滚动(334天)")
    lines.append("")

    n_pass = sum(1 for s in lines if s.startswith("[PASS]"))
    n_fail = sum(1 for s in lines if s.startswith("[FAIL]"))
    lines.append("===== 校验完成：%d 通过 / %d 失败 =====" % (n_pass, n_fail))

    open("_validation.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
