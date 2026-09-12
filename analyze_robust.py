# -*- coding: utf-8 -*-
"""P2 扩展：箱式不确定集下的**鲁棒优化**（最坏情况视角，补全"期望/风险/最坏"三视角）。

不确定集（只考虑不利方向）：
    PV_t ∈ [ PV̂_t − γ·σ_t ,  PV̂_t ] ，  γ ≥ 0 为保守度
其中 σ_t 为按 (月, 小时) 估计的历史预报误差标准差（附件三预报 − 附件二实际）。

**关键简化（严格）**：总成本对光伏单调递减（光伏越低 → 缺口越大 → 紧急购电越多），
故不确定集内的**最坏实现就是下界** PV^min = max(0, PV̂ − γσ)。于是鲁棒问题
    min_G  Σ_t p_t G_t + 5 Σ_t p_t e_t(G, PV^min)
化为一组**确定性两阶段问题**，可用已有的全局 LP + 精确追索直接求解，无需对偶化。

  γ = 0 退化为点预报计划（§5.2 的鲁棒方案）；γ 越大越保守。
  （箱式集是 Bertsimas–Sim Γ-预算集的"最保守"特例 Γ = 时段数；预算集的保守度介于
   箱式与点预报之间，可作为后续工作。）

用法：python analyze_robust.py [gamma ...]     缺省 0 0.5 1 1.5 2
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, load_fj2
from common import build_pv_forecast_stage
import solve_q4
import analyze_stochastic as A

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
T0 = A.T0_DAY


def build_sigma(err, DATES):
    """按 (月, 小时) 估计预报误差标准差 → (365,144) 矩阵。"""
    months = np.array([A._month(d) for d in DATES])
    sig = np.zeros((13, 24))
    for m in range(1, 13):
        for hh in range(24):
            vals = err[months == m][:, hh * 6:(hh + 1) * 6].ravel()
            sig[m, hh] = vals.std()
    out = np.zeros((365, 144))
    for i, d in enumerate(DATES):
        m = d.month
        for hh in range(24):
            out[i, hh * 6:(hh + 1) * 6] = sig[m, hh]
    return out


def main():
    gammas = [float(x) for x in sys.argv[1:]] or [0.0, 0.5, 1.0, 1.5, 2.0]
    price_all = np.tile(load_fj1()["price"], (365, 1))
    DATES, LOAD_all, PV_all = load_fj2()
    pv0_all = build_pv_forecast_stage()[0]
    err = pv0_all - PV_all
    sig_mat = build_sigma(err, DATES)
    p = price_all[T0:]

    # 对照：点预报计划（EV 解）
    G_ev = solve_q4._global_lp(price_all, LOAD_all, pv0_all)['G']
    plan_fee_ev = float(np.sum(G_ev[T0:] * p))
    rc_ev = solve_q4._global_recourse(G_ev, price_all, LOAD_all, PV_all)
    em_ev_act = 5.0 * float(np.sum(rc_ev['e'][T0:] * p))

    out = ["===== 箱式不确定集鲁棒优化（最坏情况视角，γ = 保守度）=====",
           "不确定集：PV ∈ [PV̂ − γσ, PV̂]；σ 为按 (月,小时) 的预报误差标准差",
           "最坏实现 = 下界 PV^min（成本对光伏单调递减），故鲁棒问题 = 确定性两阶段问题",
           "口径：365 天连续储能/首末 6000 kWh，统计 334 天", "",
           "[对照] 点预报计划：计划费 %.2f 万元；实际光伏下紧急费 %.2f 万元；合计 %.2f 万元"
           % (plan_fee_ev / 1e4, em_ev_act / 1e4, (plan_fee_ev + em_ev_act) / 1e4), ""]

    rows = []
    for g in gammas:
        pv_min = np.maximum(0.0, pv0_all - g * sig_mat)          # 最坏场景
        G = solve_q4._global_lp(price_all, LOAD_all, pv_min)['G']
        plan_fee = float(np.sum(G[T0:] * p))
        rc_w = solve_q4._global_recourse(G, price_all, LOAD_all, pv_min)
        em_w = 5.0 * float(np.sum(rc_w['e'][T0:] * p))
        rc_a = solve_q4._global_recourse(G, price_all, LOAD_all, PV_all)
        em_a = 5.0 * float(np.sum(rc_a['e'][T0:] * p))
        rows.append(dict(g=g, plan_fee=plan_fee, em_worst=em_w, total_worst=plan_fee + em_w,
                         em_act=em_a, total_act=plan_fee + em_a,
                         margin=float(np.sum(G[T0:] - G_ev[T0:]) / G[T0:].shape[0])))
        out += ["---- γ = %.1f ----" % g,
                "  计划购电费 %.2f 万元" % (plan_fee / 1e4),
                "  【最坏场景】紧急费 %.2f 万元 → 鲁棒总成本 %.2f 万元"
                % (em_w / 1e4, (plan_fee + em_w) / 1e4),
                "  【实际光伏】紧急费 %.2f 万元 → 实际总费用 %.2f 万元"
                % (em_a / 1e4, (plan_fee + em_a) / 1e4),
                "  安全裕度（相对点预报计划）= %.2f kWh/日" % rows[-1]['margin'], ""]

    # 点预报计划在"最坏场景"下的表现（用于说明鲁棒化的收益）
    rc_ev_w = solve_q4._global_recourse(G_ev, price_all, LOAD_all,
                                        np.maximum(0.0, pv0_all - gammas[-1] * sig_mat))
    ev_worst = plan_fee_ev + 5.0 * float(np.sum(rc_ev_w['e'][T0:] * p))
    last = rows[-1]
    out += ["---- 鲁棒化的价值（γ = %.1f 的最坏场景下）----" % last['g'],
            "  点预报计划的最坏场景成本 %.2f 万元 vs 鲁棒计划 %.2f 万元（省 %.2f 万元）"
            % (ev_worst / 1e4, last['total_worst'] / 1e4, (ev_worst - last['total_worst']) / 1e4),
            "  代价：实际光伏下鲁棒计划比点预报计划多付 %.2f 万元（保守性的代价）"
            % ((last['total_act'] - (plan_fee_ev + em_ev_act)) / 1e4), ""]
    open("_robust_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图 ----
    x = [r['g'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(x, [r['plan_fee'] / 1e4 for r in rows], "o-", label="计划购电费")
    ax[0].plot(x, [r['em_worst'] / 1e4 for r in rows], "s-", label="最坏场景紧急费")
    ax[0].plot(x, [r['total_worst'] / 1e4 for r in rows], "^-", label="鲁棒总成本")
    ax[0].axhline((plan_fee_ev + em_ev_act) / 1e4, ls=":", color="gray", label="点预报计划(实际)")
    ax[0].set_xlabel("保守度 $\\gamma$"); ax[0].set_ylabel("万元")
    ax[0].set_title("保守度提高：计划费升、最坏紧急费降"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot([r['total_act'] / 1e4 for r in rows], [r['total_worst'] / 1e4 for r in rows],
               "o-", color="#c0392b")
    for r in rows:
        ax[1].annotate(r"$\gamma$=%.1f" % r['g'], (r['total_act'] / 1e4, r['total_worst'] / 1e4),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_xlabel("实际光伏下的总费用（万元）"); ax[1].set_ylabel("最坏场景总成本（万元）")
    ax[1].set_title("鲁棒性—正常表现 权衡"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_鲁棒优化." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_鲁棒优化.png")


if __name__ == "__main__":
    main()
