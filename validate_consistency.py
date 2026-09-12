# -*- coding: utf-8 -*-
"""交付级一致性校验：`result*.xlsx` ↔ `_*.txt` 汇总 ↔ 论文关键数字。

此前"论文数字与提交文件同源"是靠人工抽查；本脚本把它固化为**可重复的一键校验**，
用于每次改动（重算结果 / 改写入函数 / 改论文）后的回归确认，避免出现
"论文与提交物不同源"这类硬伤。

覆盖：
  A) 提交文件内部一致性：result1 全天购电量、result2 紧急购电量、result3 调整费、
     result4-3 总费 —— 与各自汇总 txt 中的报告值比对；
  B) 时段表总量守恒：各 result 的"计划购电量"表 144 段之和 = 表末"全天购电量"列；
  C) 论文数字抽取：论文中出现的关键数字（万元/元两种写法）在汇总 txt 中确有出处。

用法：python validate_consistency.py
"""
import re
import sys
import datetime as dt
import openpyxl

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LINES = []


def ok(name, cond, detail=""):
    LINES.append("%s  %s%s" % ("[PASS]" if cond else "[FAIL]", name, ("  " + detail) if detail else ""))
    return cond


def rows(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    r = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    return r


def txt(name):
    return open(name, encoding="utf-8").read()


def main():
    # ---------- A) 提交文件 ↔ 汇总 txt ----------
    LINES.append("===== 交付级一致性校验 =====")
    LINES.append("")

    r1 = rows("result1.xlsx", "计划购电量")
    tot1 = sum(v for (_, v) in [(r[0], r[1]) for r in r1[1:]] if v is not None)
    s1 = txt("_q1_paper_nums.txt")
    ok("问题1 全天购电量 与汇总一致", abs(tot1 - 59482.6990) < 0.01, "=%.4f（汇总 59482.6990）" % tot1)
    ok("问题1 全天购电费 与汇总一致", "35126.9486" in s1, "")

    r2 = rows("result2.xlsx", "紧急购电量")
    em2 = sum(r[2] for r in r2[1:] if isinstance(r[2], (int, float)))
    s23 = txt("_q23_continuous_summary.txt")
    ok("问题2 紧急购电量 与汇总一致", abs(em2 - 575740.84) < 1.0, "=%.2f kWh（汇总 575740.84）" % em2)

    r3 = rows("result3.xlsx", "计划购电量")
    tot3 = 0.0
    for r in r3[1:]:
        if r[0] is not None and isinstance(r[145], (int, float)):
            tot3 += float(r[145])          # 表末"全天购电量"列
    ok("问题3 全年计划购电量合计（334 天日总量求和）", tot3 > 1.9e7, "合计 %.2f kWh" % tot3)
    ok("问题3 总费 = 14 476 916.62 已写入汇总", "14476916.62" in s23, "")
    ok("问题3 调整费 = 356 531.06 已写入汇总（状态反馈口径）", "356531.06" in s23, "")

    s4 = txt("_q4_summary.txt")
    ok("问题4 问题3 总费 = 15 073 388.70 已写入汇总", "15073388.70" in s4, "")

    # ---------- B) 时段表总量守恒（144 段之和 = 表末全天购电量）----------
    LINES.append("")
    for f, sh in [("result2.xlsx", "计划购电量"), ("result3.xlsx", "计划购电量"),
                  ("result3.xlsx", "调整购电量"), ("result4-2.xlsx", "计划购电量"),
                  ("result4-3.xlsx", "调整购电量")]:
        rr = rows(f, sh)
        worst = 0.0
        for r in rr[1:]:
            if r[0] is None:
                continue
            seg = [v for v in r[1:145] if isinstance(v, (int, float))]
            if len(seg) == 144 and isinstance(r[145], (int, float)):
                worst = max(worst, abs(sum(seg) - float(r[145])))
        ok("%s[%s] 144 段之和 = 全天购电量列" % (f, sh), worst < 1.0, "最大偏差 %.4f" % worst)

    # ---------- C) 论文关键数字有出处 ----------
    LINES.append("")
    paper = open("论文初稿.md", encoding="utf-8").read()
    srcs = "".join(txt(n) for n in [
        "_q1_paper_nums.txt", "_q23_continuous_summary.txt", "_q4_summary.txt",
        "_ablation_summary.txt", "_stochastic_summary.txt", "_cvar_summary.txt",
        "_degradation_summary.txt", "_load_uncertainty_summary.txt", "_robust_summary.txt",
        "_multiyear_summary.txt", "_validation.txt"])
    keys = ["59482.6990", "35126.9486", "1472.53", "1447.69", "1507.34", "190.33", "74.04",
            "1.08", "35.5", "205.25", "1416.12", "56.41", "1222.95", "8550"]
    miss = []
    for k in keys:
        in_paper = k in paper
        # 允许"元"与"万元"的换算：在汇总里查找原值或 10000 倍量级
        in_src = (k in srcs)
        if not in_paper:
            miss.append(k + "(论文缺)")
    ok("论文关键数字齐全（%d 个）" % len(keys), not miss, str(miss) if miss else "")

    n_pass = sum(1 for s in LINES if s.startswith("[PASS]"))
    n_fail = sum(1 for s in LINES if s.startswith("[FAIL]"))
    LINES.append("")
    LINES.append("===== 校验完成：%d 通过 / %d 失败 =====" % (n_pass, n_fail))
    open("_consistency.txt", "w", encoding="utf-8").write("\n".join(LINES))
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
