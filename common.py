# -*- coding: utf-8 -*-
"""共享的储能/光伏建模工具（问题2/3/4 复用）。

统一采用"全年连续储能"：单日/单段 LP 的首末电量 E0/E1 由调用方传入——连续优化时
来自全局计划 SOC 轨迹（而非固定 6000），因此本模块不含"每日复位"逻辑。

  build_pv_forecast_stage() —— 附件3 -> {0,6,12,18: (365,144)} 各时刻光伏预报
  stage_lp()               —— 滚动时域中 [t0,144) 段购电+储能 LP（含违约/溢价）
  solve_day()              —— 单日 144 段储能+购电 LP（平衡取不等式，允许弃光）
"""
import numpy as np
import datetime as dt
from scipy.optimize import linprog
from data_loader import (load_fj3, DT, PMAX_E, E_MAX, E_MIN, E_INIT, ETA, N_SLOT)

# 判断"紧急购电/调整量是否为零"的统一阈值(kWh)：高于 HiGHS 求解容差(~1e-7)、
# 低于结果文件 4 位小数精度(1e-4)，既不误判数值残差、也不丢失有效数据。
MERGE_EPS = 1e-6


def build_pv_forecast_stage():
    """附件3 -> {0,6,12,18: (365,144)} 各时刻预报(step-hold到10分钟)。
    0:00覆盖全天24h；6:00覆盖当天6-24h；12:00覆盖12-24h；18:00覆盖18-24h。"""
    fc = load_fj3()
    base = dt.datetime(2025, 1, 1)
    res = {0: np.zeros((365, N_SLOT)), 6: np.zeros((365, N_SLOT)),
           12: np.zeros((365, N_SLOT)), 18: np.zeros((365, N_SLOT))}
    for i in range(365):
        d = base + dt.timedelta(days=i)
        key = "%d-%d-%d" % (d.year, d.month, d.day)
        for tau in (0, 6, 12, 18):
            hourly = fc.get((key, "%d:00" % tau))
            if hourly is None:
                raise KeyError((key, tau))
            t0 = tau * 6
            for t in range(t0, N_SLOT):
                res[tau][i, t] = hourly[t // 6 - tau]
    return res


def stage_lp(price, load, pv_f, t0, E0, E1, G_plan_seg=None):
    """[t0,144) 段购电+储能 LP。
    G_plan_seg=None: 基础计划(0:00)，目标 min Σ price·G。
    否则: 调整阶段，目标 min Σ price·(1.5·up + 0.5·dn)，G = G_plan + up - dn。
    返回 dict: G,C,D,E(E[t0..144]), up,dn(调整阶段才有)。"""
    m = N_SLOT - t0
    pseg = price[t0:]
    Le = load[t0:] * DT
    PVe = pv_f[t0:] * DT
    has_pen = G_plan_seg is not None

    nG = nC = nD = m
    nE = m + 1
    offG, offC, offD, offE = 0, m, 2 * m, 3 * m
    if has_pen:
        offU, offN = offE + nE, offE + nE + m
        n = offN + m
    else:
        offU = offN = -1
        n = offE + nE

    c = np.zeros(n)
    if has_pen:
        c[offU:offU + m] = 1.5 * pseg
        c[offN:offN + m] = 0.5 * pseg
    else:
        c[offG:offG + m] = pseg

    Aeq, beq = [], []
    for j in range(m):                       # 储能动态
        row = np.zeros(n)
        row[offE + j + 1] = 1.0
        row[offE + j] = -1.0
        row[offC + j] = -ETA
        row[offD + j] = 1.0 / ETA
        Aeq.append(row); beq.append(0.0)
    row = np.zeros(n); row[offE + 0] = 1.0; Aeq.append(row); beq.append(E0)
    row = np.zeros(n); row[offE + m] = 1.0; Aeq.append(row); beq.append(E1)
    if has_pen:                              # G = G_plan + up - dn
        for j in range(m):
            row = np.zeros(n)
            row[offG + j] = 1.0
            row[offU + j] = -1.0
            row[offN + j] = 1.0
            Aeq.append(row); beq.append(G_plan_seg[j])

    Aub, bub = [], []                        # G + pv·Δt + D - C >= load·Δt
    for j in range(m):
        row = np.zeros(n)
        row[offG + j] = -1.0
        row[offD + j] = -1.0
        row[offC + j] = 1.0
        Aub.append(row); bub.append(PVe[j] - Le[j])

    lb = np.zeros(n); ub = np.full(n, np.inf)
    for j in range(m):
        ub[offC + j] = PMAX_E
        ub[offD + j] = PMAX_E
    for j in range(nE):
        lb[offE + j] = E_MIN; ub[offE + j] = E_MAX
    if has_pen:
        for j in range(m):
            ub[offU + j] = 1e9; ub[offN + j] = 1e9

    res = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq),
                  A_ub=np.array(Aub), b_ub=np.array(bub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    out = {'G': res.x[offG:offG + m], 'C': res.x[offC:offC + m],
           'D': res.x[offD:offD + m], 'E': res.x[offE:offE + nE]}
    if has_pen:
        out['up'] = res.x[offU:offU + m]
        out['dn'] = res.x[offN:offN + m]
    return out


def solve_day(price, load, pv, E0=E_INIT, E1=E_INIT, E_min=E_MIN, E_max=E_MAX):
    """解一天 144 段的储能+购电 LP（平衡用不等式，允许弃光）。
    price:(144,) 元/kWh；load/pv:(144,) kW。返回 dict: G,C,D,E(145),cost。
    E_min/E_max 为储能 SOC 上下限(用于敏感性测试，默认用全局 E_MIN/E_MAX)。"""
    Le = load * DT
    PVe = pv * DT
    nG = nC = nD = N_SLOT
    nE = N_SLOT + 1
    offG, offC, offD, offE = 0, nG, nG + nC, nG + nC + nD
    n = offE + nE

    c = np.zeros(n)
    c[offG:offG + nG] = price

    # 等式约束：储能动态 + 初末电量
    Aeq, beq = [], []
    for t in range(N_SLOT):
        row = np.zeros(n)
        row[offE + t + 1] = 1.0
        row[offE + t] = -1.0
        row[offC + t] = -ETA
        row[offD + t] = 1.0 / ETA
        Aeq.append(row); beq.append(0.0)
    row = np.zeros(n); row[offE + 0] = 1.0; Aeq.append(row); beq.append(E0)
    row = np.zeros(n); row[offE + N_SLOT] = 1.0; Aeq.append(row); beq.append(E1)

    # 不等式约束：G + D - C >= (L-PV)·Δt  =>  -G - D + C <= (PV-L)·Δt
    Aub, bub = [], []
    for t in range(N_SLOT):
        row = np.zeros(n)
        row[offG + t] = -1.0
        row[offD + t] = -1.0
        row[offC + t] = 1.0
        Aub.append(row); bub.append(PVe[t] - Le[t])

    lb = np.zeros(n); ub = np.full(n, np.inf)
    for t in range(N_SLOT):
        ub[offC + t] = PMAX_E
        ub[offD + t] = PMAX_E
    for t in range(nE):
        lb[offE + t] = E_min
        ub[offE + t] = E_max

    res = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq),
                  A_ub=np.array(Aub), b_ub=np.array(bub),
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return {
        'G': res.x[offG:offG + nG],
        'C': res.x[offC:offC + nC],
        'D': res.x[offD:offD + nD],
        'E': res.x[offE:offE + nE],
        'cost': float(np.dot(res.x[offG:offG + nG], price)),
    }
