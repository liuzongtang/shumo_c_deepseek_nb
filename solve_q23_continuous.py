# -*- coding: utf-8 -*-
"""问题2/3 修正版：全年连续储能（允许跨日搬移），附件1每日同价。

修正原因：问题2/3 题目并未要求"每天 24:00 回到 6000 kWh"，只要求储电量保持在
[1200,10800]。此前 solve_q2/solve_q3 强制每日复位，导致结果文件里 0:00/24:00 储电量
恒为 6000，且晴天过剩光伏无法跨日利用（全年弃光约 99 万 kWh）。

本脚本复用 solve_q4 的全局连续 LP 机制（原用于附件4波动电价），把电价改为附件1
逐日重复的分时电价，得到问题2（确定性/鲁棒）与问题3（滚动）的连续储能正确结果。

产出：
  result2.xlsx（鲁棒，连续储能）、result2_deterministic.xlsx（确定性，连续）
  result3.xlsx（滚动，连续储能）
  _q23_continuous_summary.txt（新旧对比汇总）
"""
import numpy as np
from data_loader import load_fj1, load_fj2, N_SLOT
import solve_q4
from solve_q3 import build_pv_forecast_stage


def main():
    d1 = load_fj1()
    price1 = d1['price']
    dates, load, pv = load_fj2()
    price_mat = np.tile(price1, (365, 1))          # 附件1 逐日重复

    # 复用 solve_q4 的全局函数（其内部引用模块全局 PRICE_MAT/LOAD/PV）
    solve_q4.PRICE_MAT = price_mat
    solve_q4.LOAD = load
    solve_q4.PV = pv
    solve_q4.DATES = dates

    pvf = build_pv_forecast_stage()

    # 确定性：完美预见，全年连续 LP
    det = solve_q4.run_deterministic()
    det['e'] = np.zeros((334, N_SLOT))

    # 鲁棒：全年 0:00 预报做计划 + 实际光伏重调度最小化紧急购电（连续储能）
    rob = solve_q4.run_robust(pvf[0])

    # 滚动：0/6/12/18 预报滚动调整（连续储能）
    roll = solve_q4.run_rolling(pvf)

    p = price_mat[31:]

    def rb_cost(res):
        plan = float(np.sum(res['G'] * p))
        em = float(np.sum(res['e'] * p)) * 5.0
        return plan, em, plan + em

    def rl_cost(res):
        plan = float(np.sum(res['G_plan'] * p))
        adj = float(np.sum((1.5 * res['up'] + 0.5 * res['dn']) * p))
        em = float(np.sum(res['e'] * p)) * 5.0
        return plan, adj, em, plan + adj + em

    out = []
    out.append("===== 问题2/3 修正：全年连续储能（附件1 每日同价）=====")
    out.append("")
    out.append("---- 问题2 确定性（完美预见）----")
    out.append("  连续储能: %.2f 元   (旧·每日复位: 12245046.92 元)" % det['cost'])
    out.append("")
    p2, e2, t2 = rb_cost(rob)
    out.append("---- 问题2 鲁棒（0:00 预报）----")
    out.append("  计划费: %.2f 元" % p2)
    out.append("  紧急费(5倍): %.2f 元" % e2)
    out.append("  总费: %.2f 元   (旧·每日复位: 14647581.90 元)" % t2)
    out.append("  紧急购电量: %.2f kWh, %d/334 天" % (rob['e'].sum(), (rob['e'].sum(1) > 1e-9).sum()))
    out.append("")
    p3, a3, e3, t3 = rl_cost(roll)
    out.append("---- 问题3 滚动（0/6/12/18 预报）----")
    out.append("  计划费: %.2f 元" % p3)
    out.append("  调整费: %.2f 元" % a3)
    out.append("  紧急费(5倍): %.2f 元" % e3)
    out.append("  总费: %.2f 元   (旧·每日复位: 14493184.73 元)" % t3)
    out.append("  紧急购电量: %.2f kWh, %d/334 天" % (roll['e'].sum(), (roll['e'].sum(1) > 1e-9).sum()))
    out.append("  上调天数: %d, 下调天数: %d" % ((roll['up'].sum(1) > 1e-9).sum(), (roll['dn'].sum(1) > 1e-9).sum()))
    out.append("")
    # 储电量逐日分布
    E0 = rob['E'][0::N_SLOT]
    out.append("---- 鲁棒方案 0:00 储电量分布 ----")
    out.append("  min=%.1f max=%.1f, 唯一值个数=%d, 恰为6000的天数=%d/%d"
               % (E0.min(), E0.max(), len(np.unique(np.round(E0, 1))),
                  (np.abs(E0 - 6000) < 1).sum(), len(E0)))

    # 先写汇总（不依赖结果文件锁）
    open("_q23_continuous_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # 再写结果文件（若被 Excel 占用会 PermissionError，此时改用 _new 后缀）
    for fn, writer, res in [("result2.xlsx", solve_q4._write_result4_2, rob),
                            ("result2_deterministic.xlsx", solve_q4._write_result4_2, det),
                            ("result3.xlsx", solve_q4._write_result4_3, roll)]:
        try:
            writer(res, fn)
        except PermissionError:
            alt = fn.replace(".xlsx", "_new.xlsx")
            writer(res, alt)
            out.append("!! %s 被占用，已写到 %s" % (fn, alt))
    print("done")


if __name__ == "__main__":
    main()
