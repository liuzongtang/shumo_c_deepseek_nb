# -*- coding: utf-8 -*-
"""问题4 波动电价场景深化（方向 A/B/C）。

A 鲁棒层储能价值三分解：套利 + 消纳 + 兜底误差。
B 紧急购电的电价×误差耦合：量贡献 vs 价贡献、日内时段分布。
C 跨日套利微小的对偶证明：日内价差 vs 日间价差、ψ 日内波动 vs 日间均值、
  每日复位 vs 连续 LP（跨日套利价值）。

口径：与 §6.3/§6.5 一致，E_MIN=1200、E_INIT=6000；确定性对偶解复用 analyze_dual_ext.solve；
鲁棒层复用 solve_q4._global_lp / _global_recourse（全年连续储能）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import (load_fj1, load_fj2, load_fj4, DT, ETA, E_MIN, E_MAX, E_INIT,
                         N_SLOT)
from common import build_pv_forecast_stage, solve_day
from analyze_dual_ext import solve, storage_value_split
import solve_q4 as q4

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

SL = slice(31, 365)          # 报告期 2.1–12.31（334 天）
W = 1e4                      # 元 -> 万元


def no_storage_perfect(price_mat, load_mat, pv_mat):
    """完美预见无储能：逐段买实际缺口。返回 334 天总费。"""
    gap = np.maximum(0.0, load_mat - pv_mat) * DT
    return float(np.sum(gap[SL] * price_mat[SL]))


def no_storage_robust(price_mat, load_mat, pv0, pv_act):
    """仅 0:00 预报无储能：计划买预报缺口 + 5 倍紧急补实际缺口。返回 (总费,计划费,紧急费,e)。"""
    G_plan = np.maximum(0.0, load_mat - pv0) * DT
    gap_act = np.maximum(0.0, load_mat - pv_act) * DT
    e = np.maximum(0.0, gap_act - G_plan)
    plan_fee = float(np.sum(G_plan[SL] * price_mat[SL]))
    emer_fee = float(5.0 * np.sum(e[SL] * price_mat[SL]))
    return plan_fee + emer_fee, plan_fee, emer_fee, e[SL]


def robust_storage(price_mat, load_mat, pv0, pv_act):
    """有储能鲁棒（全年连续）：0:00 预报计划 + 实际光伏重调度。返回 (总费,计划费,紧急费,e)。"""
    plan = q4._global_lp(price_mat, load_mat, pv0)
    rc = q4._global_recourse(plan['G'], price_mat, load_mat, pv_act)
    G = plan['G'][SL]; e = rc['e'][SL]; p = price_mat[SL]
    plan_fee = float(np.sum(G * p))
    emer_fee = float(5.0 * np.sum(e * p))
    return plan_fee + emer_fee, plan_fee, emer_fee, e


def main():
    d1 = load_fj1(); price1 = d1['price']
    _, LOAD, PV = load_fj2()
    _, price4 = load_fj4()
    pm_flat = np.tile(price1, (365, 1))
    pvf = build_pv_forecast_stage()
    pv0 = pvf[0]

    # 确定性对偶解（复用 analyze_dual_ext.solve）
    r_flat = solve(pm_flat, LOAD, PV, E_MAX, 5000.0)
    r_vol = solve(price4, LOAD, PV, E_MAX, 5000.0)

    out = []
    out.append("===== 问题4 波动电价深化 A/B/C =====")
    out.append("口径：E_MIN=1200、E_INIT=6000；确定性对偶=全年 365 天 LP；费用均报告期 334 天。")
    out.append("")

    # ================= 方向 A：鲁棒层储能价值三分解 =================
    out.append("【A 鲁棒层储能价值三分解：套利 + 消纳 + 兜底误差】")
    # 完美预见层
    c_no_pf_f = no_storage_perfect(pm_flat, LOAD, PV)
    c_no_pf_v = no_storage_perfect(price4, LOAD, PV)
    c_yes_pf_f = r_flat['cost334']
    c_yes_pf_v = r_vol['cost334']
    # 鲁棒层（仅 0:00 预报）
    c_no_rb_f, pn_f, en_f, e_no_f = no_storage_robust(pm_flat, LOAD, pv0, PV)
    c_no_rb_v, pn_v, en_v, e_no_v = no_storage_robust(price4, LOAD, pv0, PV)
    c_yes_rb_f, py_f, ey_f, e_f = robust_storage(pm_flat, LOAD, pv0, PV)
    c_yes_rb_v, py_v, ey_v, e_v = robust_storage(price4, LOAD, pv0, PV)

    # 确定性储能价值 = 套利 + 消纳
    v_f, a_f, ab_f = storage_value_split(pm_flat, LOAD, PV, r_flat)
    v_v, a_v, ab_v = storage_value_split(price4, LOAD, PV, r_vol)
    # 鲁棒储能价值
    V_rob_f = c_no_rb_f - c_yes_rb_f
    V_rob_v = c_no_rb_v - c_yes_rb_v
    # 兜底误差价值 = 鲁棒储能价值 - 确定性储能价值
    hedge_f = V_rob_f - v_f
    hedge_v = V_rob_v - v_v
    # 误差代价（无储能 vs 有储能）
    errcost_no_f = c_no_rb_f - c_no_pf_f
    errcost_no_v = c_no_rb_v - c_no_pf_v
    errcost_yes_f = c_yes_rb_f - c_yes_pf_f
    errcost_yes_v = c_yes_rb_v - c_yes_pf_v

    out.append("  完美预见层（无储能 -> 有储能 = 确定性储能价值）：")
    out.append("    同价   ：无储能 %.2f 万 -> 有储能 %.2f 万 = %.2f 万（=套利%.2f+消纳%.2f，校验差 %.4f 万）"
               % (c_no_pf_f / W, c_yes_pf_f / W, v_f / W, a_f / W, ab_f / W,
                  (v_f - (a_f + ab_f)) / W))
    out.append("    波动价 ：无储能 %.2f 万 -> 有储能 %.2f 万 = %.2f 万（=套利%.2f+消纳%.2f，校验差 %.4f 万）"
               % (c_no_pf_v / W, c_yes_pf_v / W, v_v / W, a_v / W, ab_v / W,
                  (v_v - (a_v + ab_v)) / W))
    out.append("  鲁棒层（无储能 -> 有储能 = 鲁棒储能价值）：")
    out.append("    同价   ：无储能 %.2f 万 -> 有储能 %.2f 万 = %.2f 万"
               % (c_no_rb_f / W, c_yes_rb_f / W, V_rob_f / W))
    out.append("    波动价 ：无储能 %.2f 万 -> 有储能 %.2f 万 = %.2f 万"
               % (c_no_rb_v / W, c_yes_rb_v / W, V_rob_v / W))
    out.append("  三分解（鲁棒储能价值 = 套利 + 消纳 + 兜底误差）：")
    out.append("    同价   ：套利 %.2f + 消纳 %.2f + 兜底 %.2f = %.2f 万"
               % (a_f / W, ab_f / W, hedge_f / W, V_rob_f / W))
    out.append("    波动价 ：套利 %.2f + 消纳 %.2f + 兜底 %.2f = %.2f 万"
               % (a_v / W, ab_v / W, hedge_v / W, V_rob_v / W))
    out.append("  兜底误差价值 = 无储能误差代价 - 有储能误差代价：")
    out.append("    同价   ：无储能误差 %.2f - 有储能误差 %.2f = %.2f 万"
               % (errcost_no_f / W, errcost_yes_f / W, hedge_f / W))
    out.append("    波动价 ：无储能误差 %.2f - 有储能误差 %.2f = %.2f 万"
               % (errcost_no_v / W, errcost_yes_v / W, hedge_v / W))
    out.append("  → 波动价下兜底误差价值 %.2f 万 vs 同价 %.2f 万（%+.2f 万）"
               % (hedge_v / W, hedge_f / W, (hedge_v - hedge_f) / W))
    out.append("")

    # ================= 方向 B：紧急购电 电价×误差 耦合 =================
    out.append("【B 紧急购电的电价×误差耦合】")
    # 紧急量、加权均价、全时段均价、紧急费
    p_vol = price4[SL]; p_flt = pm_flat[SL]
    def emer_stats(e, p):
        s = float(e.sum())
        if s <= 0:
            return 0.0, 0.0, 0.0
        wavg = float(np.sum(p * e) / s)
        avg = float(p.mean())
        fee = float(5.0 * np.sum(p * e))
        return s, wavg, avg, fee
    es_f, wa_f, av_f, ef_f = emer_stats(e_f, p_flt)
    es_v, wa_v, av_v, ef_v = emer_stats(e_v, p_vol)
    hour_e_v = e_v.sum(axis=0).reshape(24, 6).sum(axis=1)
    peak_h = int(np.argmax(hour_e_v))
    morning_frac = float(hour_e_v[5:8].sum() / es_v)
    out.append("  紧急购电量（kWh）：同价 %.2f 万 / 波动价 %.2f 万（%+.1f%%）"
               % (es_f / W, es_v / W, 100 * (es_v / es_f - 1)))
    out.append("  紧急购电加权均价（元/kWh）：同价 %.4f / 波动价 %.4f"
               % (wa_f, wa_v))
    out.append("  全时段平均电价（元/kWh）：同价 %.4f / 波动价 %.4f"
               % (av_f, av_v))
    out.append("  → 紧急购电集中在电价 %.2f%%（同价）/ %.2f%%（波动价）高于全时段均价的时段"
               % (100 * (wa_f / av_f - 1), 100 * (wa_v / av_v - 1)))
    out.append("  紧急购电费（万元）：同价 %.2f / 波动价 %.2f（贵 %.2f 万）"
               % (ef_f / W, ef_v / W, (ef_v - ef_f) / W))
    out.append("  紧急购电日内分布（波动价）：峰值 %d:00，清晨 5–7 时占 %.1f%%（光伏预报偏乐观、谷段电价）"
               % (peak_h, 100 * morning_frac))
    # 量贡献 vs 价贡献（相对全时段均价）
    qty_f = 5.0 * es_f * av_f; prc_f = ef_f - qty_f
    qty_v = 5.0 * es_v * av_v; prc_v = ef_v - qty_v
    out.append("  量×价分解（紧急费 = 量×全时段均价 + 时段选择溢价）：")
    out.append("    同价   ：量贡献 %.2f + 价贡献 %.2f = %.2f 万"
               % (qty_f / W, prc_f / W, ef_f / W))
    out.append("    波动价 ：量贡献 %.2f + 价贡献 %.2f = %.2f 万"
               % (qty_v / W, prc_v / W, ef_v / W))
    # 贵 59 万的结构来源：计划费 vs 紧急费
    out.append("  波动价比同价贵 59 万的结构来源：")
    out.append("    计划费：同价 %.2f 万 -> 波动价 %.2f 万（%+.2f 万）"
               % (py_f / W, py_v / W, (py_v - py_f) / W))
    out.append("    紧急费：同价 %.2f 万 -> 波动价 %.2f 万（%+.2f 万）"
               % (ey_f / W, ey_v / W, (ey_v - ey_f) / W))
    out.append("    → 贵 59 万主要来自计划费（跨日价差抬高了计划购电电价），紧急费仅占小头。")
    out.append("")

    # ================= 方向 C：跨日套利微小的对偶证明 =================
    out.append("【C 跨日套利微小的对偶证明】")
    # 附件4 日内 vs 日间价差
    daily_range = (price4.max(axis=1) - price4.min(axis=1))[SL]   # 每天峰谷差
    daily_mean = price4[SL].mean(axis=1)                          # 每天均价
    interday_diff = np.abs(np.diff(daily_mean))                   # 相邻日均价差
    out.append("  附件4 价差（334 天）：")
    out.append("    日内峰谷差：均值 %.4f 元/kWh（最小 %.4f / 最大 %.4f）"
               % (daily_range.mean(), daily_range.min(), daily_range.max()))
    out.append("    日间均价差（相邻天）：均值 %.4f 元/kWh（最小 %.4f / 最大 %.4f）"
               % (interday_diff.mean(), interday_diff.min(), interday_diff.max()))
    out.append("    → 日内价差/日间价差 = %.1f 倍（套利主战场是日内峰谷）"
               % (daily_range.mean() / max(interday_diff.mean(), 1e-9)))
    # ψ 日内波动 vs 日间均值差异（波动价，334 天）
    psi_vol = r_vol['psi'][SL]                     # (334,144)
    intraday_psi = psi_vol.std(axis=1).mean()      # 每天内 ψ std 均值
    daily_psi_mean = psi_vol.mean(axis=1)          # 每天 ψ 均值
    interday_psi = daily_psi_mean.std()            # 日间 ψ 均值差异
    out.append("  水值 ψ（波动价，334 天）：")
    out.append("    日内波动（每天内 std 均值）：%.4f 元/kWh" % intraday_psi)
    out.append("    日间均值差异（每天 ψ 均值的 std）：%.4f 元/kWh" % interday_psi)
    out.append("    → ψ 日内波动/日间差异 = %.1f 倍（储能的边际价值几乎全来自日内）"
               % (intraday_psi / max(interday_psi, 1e-9)))
    # 每日复位 vs 连续（跨日套利价值）
    reset_cost = 0.0
    for d in range(31, 365):
        sd = solve_day(price4[d], LOAD[d], PV[d], E0=E_INIT, E1=E_INIT)
        reset_cost += sd['cost']
    cross_day = reset_cost - r_vol['cost334']
    out.append("  跨日套利价值（确定性层）：每日复位总费 %.2f 万 - 连续总费 %.2f 万 = %.2f 万"
               % (reset_cost / W, r_vol['cost334'] / W, cross_day / W))
    out.append("    → 允许跨日搬移仅多省 %.2f 万，储能价值几乎全部来自日内峰谷套利。" % (cross_day / W))
    out.append("")

    open("_q4_deep_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ================= 图 18：A 储能价值三分解 =================
    x = np.arange(2)
    wb = 0.5
    arb = np.array([a_f, a_v]) / W
    absorb = np.array([ab_f, ab_v]) / W
    hedge = np.array([hedge_f, hedge_v]) / W
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].bar(x, arb, wb, color="#e67e22", label="净套利（峰谷搬移）")
    ax[0].bar(x, absorb, wb, bottom=arb, color="#27ae60", label="净消纳弃光")
    ax[0].bar(x, hedge, wb, bottom=arb + absorb, color="#2980b9", label="兜底预报误差")
    for xi, tot in enumerate(arb + absorb + hedge):
        ax[0].text(xi, tot + 3, "%.1f" % tot, ha="center", fontsize=9, fontweight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["附件1 同价", "附件4 波动价"])
    ax[0].set_ylabel("储能价值构成（万元）")
    ax[0].set_title("(a) 鲁棒层储能价值 = 套利 + 消纳 + 兜底误差")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    xr = np.arange(2)
    wr = 0.38
    err_no = np.array([errcost_no_f, errcost_no_v]) / W
    err_yes = np.array([errcost_yes_f, errcost_yes_v]) / W
    ax[1].bar(xr - wr / 2, err_no, wr, color="#c0392b", label="无储能误差代价")
    ax[1].bar(xr + wr / 2, err_yes, wr, color="#2980b9", label="有储能误差代价")
    for xi, (n, y) in enumerate(zip(err_no, err_yes)):
        ax[1].text(xi - wr / 2, n + 2, "%.0f" % n, ha="center", fontsize=8)
        ax[1].text(xi + wr / 2, y + 2, "%.0f" % y, ha="center", fontsize=8)
        ax[1].text(xi, max(n, y) + 12, "削减 %.0f" % (n - y), ha="center", fontsize=9,
                   color="#16a085", fontweight="bold")
    ax[1].set_xticks(xr); ax[1].set_xticklabels(["附件1 同价", "附件4 波动价"])
    ax[1].set_ylabel("预报误差代价（万元）")
    ax[1].set_title("(b) 储能削减预报误差代价（兜底价值来源）")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_储能价值三分解." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ================= 图 19：B 紧急购电 电价×误差 耦合 =================
    # (a) 紧急购电日内分布（按小时）+ 平均电价
    hour_e_v = e_v.sum(axis=0).reshape(24, 6).sum(axis=1)      # 波动价 24h 紧急量
    hour_p_v = p_vol.mean(axis=0).reshape(24, 6).mean(axis=1)  # 波动价 24h 平均电价
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].bar(np.arange(24), hour_e_v, color="#c0392b", alpha=0.75, label="紧急购电量（波动价）")
    ax[0].set_xlabel("日内时刻（时）")
    ax[0].set_ylabel("紧急购电量（kWh，334 天累计）", color="#c0392b")
    ax0b = ax[0].twinx()
    ax0b.plot(np.arange(24), hour_p_v, "o-", color="#2980b9", lw=2, label="平均电价")
    ax0b.set_ylabel("平均电价（元/kWh）", color="#2980b9")
    ax[0].set_title("(a) 紧急购电集中于清晨 5–7 时（光伏预报偏乐观）")
    ax[0].set_xticks(np.arange(0, 24, 2))
    ax[0].grid(alpha=0.3)

    # (b) 量×价分解
    qty = np.array([qty_f, qty_v]) / W
    prc = np.array([prc_f, prc_v]) / W
    x = np.arange(2)
    ax[1].bar(x, qty, 0.5, color="#95a5a6", label="量贡献（按全时段均价）")
    ax[1].bar(x, prc, 0.5, bottom=qty, color="#e67e22", label="价贡献（时段选择溢价）")
    for xi, (qq, pp) in enumerate(zip(qty, prc)):
        ax[1].text(xi, qq + pp + 2, "%.1f" % (qq + pp), ha="center", fontsize=9, fontweight="bold")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["附件1 同价", "附件4 波动价"])
    ax[1].set_ylabel("紧急购电费（万元）")
    ax[1].set_title("(b) 紧急购电费的量×价分解")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_紧急购电量价耦合." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ================= 图 20：C 跨日套利微小的对偶证明 =================
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].boxplot([daily_range, interday_diff], labels=["日内峰谷价差", "日间均价差"],
                  widths=0.5, showfliers=True)
    ax[0].set_ylabel("价差（元/kWh）")
    ax[0].set_title("(a) 附件4：日内价差远大于日间价差")
    ax[0].grid(alpha=0.3, axis="y")

    cats = ["ψ 日内波动\n（每天内 std 均值）", "ψ 日间差异\n（每天均值 std）"]
    ax[1].bar(cats, [intraday_psi, interday_psi], 0.5,
              color=["#8e44ad", "#bdc3c7"])
    for xi, val in enumerate([intraday_psi, interday_psi]):
        ax[1].text(xi, val + 0.01, "%.4f" % val, ha="center", fontsize=10, fontweight="bold")
    ax[1].set_ylabel("水值 ψ 波动幅度（元/kWh）")
    ax[1].set_title("(b) 水值 ψ 几乎全来自日内（跨日套利空间小）")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_跨日套利对偶证明." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("done; cross_day=%.2f万, hedge_v=%.2f万, hedge_f=%.2f万"
          % (cross_day / W, hedge_v / W, hedge_f / W))


if __name__ == "__main__":
    main()
