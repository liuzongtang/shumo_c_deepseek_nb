# -*- coding: utf-8 -*-
"""C题 数据读取模块
统一把附件1-4读成 numpy 数组，消除时间标签/粒度差异。
约定：
  - 一天 144 个 10 分钟段，下标 t=0..143 对应 [t*10min, (t+1)*10min]。
  - 功率(kW) 在需要时 ×(1/6) 转能量(kWh)。
"""
import numpy as np
import pandas as pd
import openpyxl
from pathlib import Path

BASE = Path(__file__).parent
ATT = BASE / "附件"

DT = 1.0 / 6.0          # 每段 10 分钟 = 1/6 小时
N_SLOT = 144            # 一天段数
PMAX = 5000.0           # 储能最大功率 kW
PMAX_E = PMAX * DT      # 每段最大充/放电量 kWh (833.33)
E_MAX = 10800.0         # 储能电量上限 kWh
E_MIN = 1200.0          # 储能电量下限 kWh
E_INIT = 6000.0         # 2025-1-1 0:00 初始电量 kWh
ETA = 0.9               # 充放电效率


def _read_row_per_slot(path, sheet=0):
    """读一个 366x145 的表(日期 + 144段)，返回 (dates, mat[N,144])。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if isinstance(sheet, int):
        ws = wb.worksheets[sheet]
    else:
        ws = wb[sheet]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    # 第0列是日期，第1..144列是 144 个时段
    assert header[0] in ("日期\\时间", "日期") or "日期" in str(header[0]), header[0]
    dates, mat = [], []
    for r in rows:
        if r[0] is None:
            continue
        dates.append(r[0])
        mat.append([float(x) for x in r[1:1 + N_SLOT]])
    wb.close()
    return dates, np.array(mat)


def load_fj1():
    """附件1：问题1用的单日数据。
    返回 dict: price(144), load(144), pv(144)  [kW/kWh]"""
    wb = openpyxl.load_workbook(ATT / "附件1.xlsx", read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    price = np.array([r[idx['电价']] for r in rows[1:]])
    load = np.array([r[idx['小区负载']] for r in rows[1:]])
    pv = np.array([r[idx['光伏发电预测功率']] for r in rows[1:]])
    assert price.size == N_SLOT, price.size
    return {'price': price, 'load': load, 'pv': pv}


def load_fj2():
    """附件2：全年实际负载/光伏。返回 (dates, load[N,144], pv[N,144])，N=365。"""
    dates, load = _read_row_per_slot(ATT / "附件2.xlsx", sheet="小区负载")
    _, pv = _read_row_per_slot(ATT / "附件2.xlsx", sheet="光伏发电实际功率")
    return dates, load, pv


def load_fj4():
    """附件4：全年电价。返回 (dates, price[N,144])。"""
    dates, price = _read_row_per_slot(ATT / "附件4.xlsx", sheet=0)
    return dates, price


def load_fj3():
    """附件3：全年光伏预报。
    返回 dict: forecast[(date_idx, hour_idx)] = array(24)，hour_idx in {0,6,12,18}。
    预报值单位 kW，每小时 1 个点(整点)，共 24 个。
    注意日期列只在每天第一条有值，需前向填充。"""
    wb = openpyxl.load_workbook(ATT / "附件3.xlsx", read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = rows[0]
    assert header[1] == '预报时刻'
    fc = {}   # (date_str, '0:00') -> np.array(24)
    cur_date = None
    for r in rows[1:]:
        if r[0] not in (None, ''):
            cur_date = str(r[0])
        t = r[1]
        vals = np.array([float(x) for x in r[2:2 + 24]])
        fc[(cur_date, t)] = vals
    return fc


if __name__ == "__main__":
    d1 = load_fj1()
    print("附件1 price/load/pv:", d1['price'].shape)
    dates, load, pv = load_fj2()
    print("附件2 load/pv:", load.shape, pv.shape, "天数", len(dates), dates[0], "->", dates[-1])
    dates4, price = load_fj4()
    print("附件4 price:", price.shape, "天数", len(dates4))
    fc = load_fj3()
    print("附件3 预报条目数:", len(fc), "应有 365*4=", 365 * 4)
    k = list(fc.keys())[:2]
    print("示例 key:", k, "样例值前6:", fc[k[0]][:6])
