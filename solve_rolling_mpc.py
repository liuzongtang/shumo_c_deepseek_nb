# -*- coding: utf-8 -*-
"""问题3 滚动层升级：状态反馈 + 24 小时前瞻窗（closed-loop MPC 口径）。

题目给定：0:00/6:00/12:00/18:00 各发布一次**未来 24 小时**整点光伏预报。
附件3 每个发布时刻都提供 24 个整点值（当天发布时刻 → 次日同一时刻）。

本脚本对照三种滚动实现（其余部分完全一致）：

  A 旧·预报驱动 / 当天窗口   —— solve_q4.run_rolling 的原口径
     6/12/18 的初始 SOC 取"预报计划轨迹"，滚动窗口截止到当天 24:00。

  B 状态反馈 / 当天窗口      —— 改法1
     6/12/18 的初始 SOC 由**实际光伏**递推得到（上午实际光照决定中午的真实电量）。

  C 状态反馈 + 24h 前瞻窗    —— 改法1+2
     窗口扩到 [t0, t0+24h)（跨日，吃满附件3 的次日预报）。窗口内当天段仍与
     0:00 计划比较计违约/溢价；次日段只计纯购电费（次日 0:00 会重新决策，
     故次日段的购电量不纳入报告）。终端电量锚定 0:00 预报计划轨迹在窗口末端的值，
     避免"为省钱掏空储能"的短视解。

执行阶段的储能规则（_advance_actual）：
  储能严格执行已下发的计划充放电；实际光伏偏差由"盈余充入储能 / 缺口紧急购电"吸收。

不改任何 result*.xlsx，只输出对照到 _rolling_mpc_summary.txt。
"""
import sys
import datetime as dt
import numpy as np
from scipy.optimize import linprog
from data_loader import DT, PMAX_E, E_MAX, E_MIN, ETA, N_SLOT, load_fj3
from common import build_pv_forecast_stage, MERGE_EPS
import solve_q4

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HOURS = (0, 6, 12, 18)


def build_hourly_forecast():
    """返回 {(day_idx, tau): np.array(24)}，直接来自附件3 的整点预报。"""
    fc = load_fj3()
    base = dt.datetime(2025, 1, 1)
    H = {}
    for d in range(365):
        day = base + dt.timedelta(days=d)
        key = "%d-%d-%d" % (day.year, day.month, day.day)
        for tau in HOURS:
            H[(d, tau)] = fc[(key, "%d:00" % tau)]
    return H


def _cross_day(mat, d, t0, m):
    """取 [t0, t0+m) 的跨日数据（段索引可超过 144）。末日越界用当天数据占位。"""
    if t0 + m <= N_SLOT:
        return mat[d][t0:t0 + m]
    head = mat[d][t0:]
    tail_len = t0 + m - N_SLOT
    nxt = mat[min(d + 1, mat.shape[0] - 1)]
    return np.concatenate([head, nxt[:tail_len]])


def _forecast_window(H, d, tau, t0, m):
    """按发布时刻 tau 的 24 小时预报填出窗口预报（step-hold 到 10 分钟段）。"""
    hourly = H[(d, tau)]
    out = np.zeros(m)
    for j in range(m):
        k = (t0 + j) // 6 - tau
        if 0 <= k <= 23:
            out[j] = hourly[k]
    return out


def _window_lp(price_w, load_w, pv_w, E0, E1, G_plan_w):
    """滚动窗口 LP。
    G_plan_w: 长度 n_day（当天剩余段）的 0:00 计划购电量；窗口后段（次日）无罚金、只计纯购电费。
    返回窗口内 G/C/D/E（E 长度 m+1）。"""
    m = len(price_w)
    n_day = len(G_plan_w)
    offG, offC, offD = 0, m, 2 * m
    offE = 3 * m
    offU = offE + m + 1
    offN = offU + n_day
    n = offN + n_day

    c = np.zeros(n)
    if m > n_day:                                        # 次日段：纯购电费（1.0 倍）
        c[offG + n_day:offG + m] = price_w[n_day:]
    c[offU:offU + n_day] = 1.5 * price_w[:n_day]         # 上调溢价
    c[offN:offN + n_day] = 0.5 * price_w[:n_day]         # 下调违约

    Aeq, beq = [], []
    for j in range(m):                                   # SOC 动态
        row = np.zeros(n)
        row[offE + j + 1] = 1.0
        row[offE + j] = -1.0
        row[offC + j] = -ETA
        row[offD + j] = 1.0 / ETA
        Aeq.append(row); beq.append(0.0)
    row = np.zeros(n); row[offE] = 1.0; Aeq.append(row); beq.append(E0)
    row = np.zeros(n); row[offE + m] = 1.0; Aeq.append(row); beq.append(E1)
    for j in range(n_day):                               # G = G_plan + up - dn
        row = np.zeros(n)
        row[offG + j] = 1.0
        row[offU + j] = -1.0
        row[offN + j] = 1.0
        Aeq.append(row); beq.append(G_plan_w[j])

    Aub, bub = [], []
    for j in range(m):                                   # G + D - C >= (L-PV)*dt
        row = np.zeros(n)
        row[offG + j] = -1.0
        row[offD + j] = -1.0
        row[offC + j] = 1.0
        Aub.append(row); bub.append(pv_w[j] * DT - load_w[j] * DT)

    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[offC:offC + m] = PMAX_E
    ub[offD:offD + m] = PMAX_E
    lb[offE:offE + m + 1] = E_MIN
    ub[offE:offE + m + 1] = E_MAX

    res = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq),
                  A_ub=np.array(Aub), b_ub=np.array(bub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return {'G': res.x[offG:offG + m], 'C': res.x[offC:offC + m],
            'D': res.x[offD:offD + m], 'E': res.x[offE:offE + m + 1]}


def _advance_actual(pv_act, load, G_seg, C_seg, D_seg, E0):
    """按实际光伏推进 SOC：储能执行计划充放电，盈余充入储能、缺口留待紧急购电。"""
    E = float(E0)
    extra = 0.0
    for t in range(len(G_seg)):
        bal = G_seg[t] + pv_act[t] * DT - load[t] * DT - C_seg[t] + D_seg[t]
        c_extra = 0.0
        if bal > 0.0:
            room_pow = PMAX_E - C_seg[t]
            room_cap = (E_MAX - E) / ETA - C_seg[t]
            c_extra = max(0.0, min(bal, room_pow, room_cap))
        E = E + ETA * (C_seg[t] + c_extra) - D_seg[t] / ETA
        extra += c_extra
        E = min(max(E, E_MIN), E_MAX)
    return E, extra


def run_rolling_mpc(H, pvf, use_window=True, feedback=True, E_plan=None):
    """滚动求解。use_window: 是否扩到 24h 跨日窗；feedback: 是否用实际 SOC 做初值。"""
    if E_plan is None:
        E_plan = solve_q4._global_lp(solve_q4.PRICE_MAT, solve_q4.LOAD, pvf[0])['E']
    ndays = 365
    G_plan = np.zeros((ndays, N_SLOT))
    G_adj = np.zeros((ndays, N_SLOT))
    extra_charged = 0.0

    for d in range(ndays):
        load_d = solve_q4.LOAD[d]
        pv_act = solve_q4.PV[d]
        E_cur = E_plan[d * N_SLOT]                     # 当天 0:00 电量
        plan_done = False

        for tau, t0 in zip(HOURS, (0, 36, 72, 108)):
            n_day = N_SLOT - t0                        # 当天剩余段数
            m = n_day + (tau * 6 if use_window else 0)  # 24h 窗时恒为 144 段
            price_w = _cross_day(solve_q4.PRICE_MAT, d, t0, m)
            load_w = _cross_day(solve_q4.LOAD, d, t0, m)
            pv_w = _forecast_window(H, d, tau, t0, m)
            gidx = min(d * N_SLOT + t0 + m, len(E_plan) - 1)
            E_t = E_plan[gidx]                         # 终端电量锚定预报计划轨迹

            if not plan_done:
                # 0:00 计划：以零为基准做"上调"，等价于纯购电费最小化
                r = _window_lp(price_w, load_w, pv_w, E_cur, E_t, np.zeros(n_day))
                G_plan[d][t0:] = r['G'][:n_day]
                plan_done = True
            else:
                r = _window_lp(price_w, load_w, pv_w, E_cur, E_t, G_plan[d][t0:])
            G_adj[d][t0:] = r['G'][:n_day]

            t_next = min(t0 + 36, N_SLOT)              # 下一个决策时点
            seg = t_next - t0
            if seg > 0 and t_next < N_SLOT:
                if feedback:                            # 状态反馈：实际光伏递推
                    E_cur, x = _advance_actual(pv_act[t0:t_next], load_d[t0:t_next],
                                               r['G'][:seg], r['C'][:seg], r['D'][:seg], E_cur)
                    extra_charged += x
                else:                                   # 预报驱动：沿用计划轨迹
                    E_cur = E_plan[d * N_SLOT + t_next]

    rc = solve_q4._global_recourse(G_adj, solve_q4.PRICE_MAT, solve_q4.LOAD, solve_q4.PV)
    up = np.maximum(0.0, G_adj - G_plan)
    dn = np.maximum(0.0, G_plan - G_adj)
    res = {'G_plan': G_plan, 'G_adj': G_adj, 'up': up, 'dn': dn,
           'C': rc['C'], 'D': rc['D'], 'E': rc['E'], 'e': rc['e'],
           'extra_charged': extra_charged}
    return solve_q4._slice_to_reported(res)


def main():
    solve_q4.DATES, solve_q4.LOAD, solve_q4.PV = solve_q4.load_fj2()
    _, solve_q4.PRICE_MAT = solve_q4.load_fj4()
    pvf = build_pv_forecast_stage()
    H = build_hourly_forecast()
    price = solve_q4.PRICE_MAT[31:]

    def cost_rl(res):
        plan = float(np.sum(res['G_plan'] * price))
        adj = float(np.sum((1.5 * res['up'] + 0.5 * res['dn']) * price))
        em = float(np.sum(res['e'] * price)) * 5.0
        return plan, adj, em, plan + adj + em

    def report(tag, res, note=""):
        p, a, e, t = cost_rl(res)
        return (["[%s]%s" % (tag, note),
                 "  计划费 %.2f + 调整费 %.2f + 紧急费 %.2f = %.2f 元" % (p, a, e, t),
                 "  紧急购电量 %.2f kWh, 上调天数 %d, 下调合计 %.3e"
                 % (res['e'].sum(), (res['up'].sum(1) > MERGE_EPS).sum(), res['dn'].sum()),
                 "  SOC 范围 [%.2f, %.2f]" % (res['E'].min(), res['E'].max()),
                 "  执行期额外回充光伏 %.2f kWh" % res.get('extra_charged', 0.0), ""], t)

    out = ["===== 问题3 滚动层：三种实现对照（附件4 波动电价口径）=====", ""]

    ra = report("A 旧·预报驱动/当天窗口", solve_q4.run_rolling(pvf), "  <- 现论文口径")
    rb = report("B 状态反馈/当天窗口", run_rolling_mpc(H, pvf, use_window=False, feedback=True))
    rc = report("C 状态反馈+24h前瞻窗", run_rolling_mpc(H, pvf, use_window=True, feedback=True))
    out += ra[0] + rb[0] + rc[0]
    tA, tB, tC = ra[1], rb[1], rc[1]
    out += ["改进对照:",
            "  B - A = %+.2f 元  (状态反馈收益)" % (tB - tA),
            "  C - B = %+.2f 元  (24h 前瞻窗收益)" % (tC - tB),
            "  C - A = %+.2f 元  (合计)" % (tC - tA)]
    open("_rolling_mpc_summary.txt", "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
