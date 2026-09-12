# -*- coding: utf-8 -*-
"""方向2：光伏预报误差 ↔ 紧急购电 的定量关联。

机理：紧急购电发生在 0:00 预报"高估"光伏（预报>实际）导致计划少购、且储能不足以
兜底时。因此"逐日高估量"应与"逐日紧急购电量"强相关。量化该关联、并分月汇总。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_loader import load_fj1, load_fj2, DT
import solve_q4
from common import build_pv_forecast_stage

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def _month(d):
    return getattr(d, "month", None) or dt.datetime.strptime(str(d)[:10], "%Y-%m-%d").month


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    return float((xm * ym).sum() / den) if den > 0 else 0.0


def main():
    d1 = load_fj1()
    price1 = d1["price"]
    DATES, LOAD, PV = load_fj2()
    price_mat = np.tile(price1, (365, 1))
    solve_q4.PRICE_MAT = price_mat
    solve_q4.LOAD = LOAD
    solve_q4.PV = PV
    solve_q4.DATES = DATES
    pvf = build_pv_forecast_stage()
    pv0 = pvf[0]

    rob = solve_q4.run_robust(pv0)
    e = rob["e"]                              # (334,144) kWh
    pv0_rep = pv0[31:]
    PV_rep = PV[31:]
    dates_rep = DATES[31:]

    err = pv0_rep - PV_rep                    # kW，预报−实际（>0 高估）
    daily_over = np.sum(np.maximum(0.0, err) * DT, axis=1)   # 日高估量 kWh
    daily_mae = np.mean(np.abs(err), axis=1)                 # kW
    daily_rmse = np.sqrt(np.mean(err ** 2, axis=1))          # kW
    daily_em = e.sum(axis=1)                                 # 日紧急购电量 kWh

    r_over = _pearson(daily_over, daily_em)
    r_mae = _pearson(daily_mae, daily_em)
    r_rmse = _pearson(daily_rmse, daily_em)

    # 分月汇总
    months = np.array([_month(d) for d in dates_rep])
    out = []
    out.append("===== 方向2 光伏预报误差 ↔ 紧急购电 定量关联 =====")
    out.append("逐日皮尔逊相关：")
    out.append("  高估量 vs 紧急购电   r = %+.3f" % r_over)
    out.append("  MAE     vs 紧急购电   r = %+.3f" % r_mae)
    out.append("  RMSE    vs 紧急购电   r = %+.3f" % r_rmse)
    out.append("")
    out.append("分月汇总（高估量 kWh / 紧急购电 kWh）：")
    mm = sorted(set(months.tolist()))
    over_m, em_m = [], []
    for m in mm:
        mk = months == m
        om = float(daily_over[mk].sum())
        emm = float(daily_em[mk].sum())
        over_m.append(om); em_m.append(emm)
        out.append("  %2d月  高估 %.0f kWh  紧急 %.0f kWh" % (m, om, emm))
    open("_error_emergency_summary.txt", "w", encoding="utf-8").write("\n".join(out))

    # ---- 图：(a) 分月柱状 (b) 逐日散点+拟合 ----
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(len(mm))
    w = 0.4
    ax[0].bar(x - w / 2, np.array(over_m) / 1e3, w, label="光伏预报高估量", color="#e67e22")
    ax[0].bar(x + w / 2, np.array(em_m) / 1e3, w, label="紧急购电量", color="#e74c3c")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["%d月" % m for m in mm])
    ax[0].set_ylabel("电量（千 kWh）")
    ax[0].set_title("(a) 分月：预报高估量与紧急购电量")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].scatter(daily_over / 1e3, daily_em / 1e3, s=12, alpha=0.5, color="#2980b9")
    k = np.polyfit(daily_over, daily_em, 1)
    xs = np.linspace(daily_over.min(), daily_over.max(), 50)
    ax[1].plot(xs / 1e3, np.polyval(k, xs) / 1e3, color="#c0392b", lw=1.6,
               label="线性拟合 r=%+.2f" % r_over)
    ax[1].set_xlabel("日光伏预报高估量（千 kWh）")
    ax[1].set_ylabel("日紧急购电量（千 kWh）")
    ax[1].set_title("(b) 逐日：高估量 vs 紧急购电量")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("fig_误差与紧急购电." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(out))
    print("已生成 figures/fig_误差与紧急购电.png")


if __name__ == "__main__":
    main()
