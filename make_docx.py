# -*- coding: utf-8 -*-
"""一键生成 `论文初稿.docx`：pandoc 转换 + 中英文字体定制。

流程：
  1. `pandoc 论文初稿.md -o 论文初稿.docx`（LaTeX 公式自动转为 Word 原生 OMML）；
  2. 解压 docx，把 `word/styles.xml` 中的 theme 字体（asciiTheme/eastAsiaTheme/...）
     替换为显式字体——**中文宋体 + 英文 Times New Roman**（中文论文排版惯例）；
  3. 重新打包（保留全部内嵌图与公式）。

用法：python make_docx.py [src.md] [dst.docx]
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile

SRC = sys.argv[1] if len(sys.argv) > 1 else "论文初稿.md"
DST = sys.argv[2] if len(sys.argv) > 2 else "论文初稿.docx"

FONT_XML = '<w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
THEME_PAT = re.compile(r'<w:rFonts[^>]*w:(?:asciiTheme|eastAsiaTheme|hAnsiTheme|cstheme)="[^"]*"[^>]*/>')


def main():
    if os.path.exists(DST):
        os.makedirs("_pre_format_backup", exist_ok=True)
        shutil.copy(DST, os.path.join("_pre_format_backup", os.path.basename(DST)))
        print("已备份旧版 ->", "_pre_format_backup/" + os.path.basename(DST))

    subprocess.run(["pandoc", SRC, "-o", DST], check=True)
    print("pandoc 转换完成 ->", DST)

    zin = zipfile.ZipFile(DST)
    names = zin.namelist()
    data = {n: zin.read(n) for n in names}
    zin.close()

    s = data["word/styles.xml"].decode("utf-8")
    n_before = s.count("w:rFonts")
    s, n_sub = THEME_PAT.subn(FONT_XML, s)
    for old in ("Cambria", "Calibri", "等线", "Microsoft YaHei"):
        s = (s.replace('w:ascii="%s"' % old, 'w:ascii="Times New Roman"')
               .replace('w:hAnsi="%s"' % old, 'w:hAnsi="Times New Roman"')
               .replace('w:eastAsia="%s"' % old, 'w:eastAsia="宋体"'))
    data["word/styles.xml"] = s.encode("utf-8")

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, data[n])

    z = zipfile.ZipFile(DST)
    xml = z.read("word/document.xml").decode("utf-8")
    print("字体定制：theme 字体替换 %d 处（rFonts %d 个）；中文宋体 %d 处"
          % (n_sub, s.count("w:rFonts"), s.count('w:eastAsia="宋体"')))
    print("校验：zip 完整性 %s | Word 公式 %d 个 | 内嵌图 %d 张 | 体积 %.2f MB"
          % ("OK" if z.testzip() is None else "损坏", xml.count("<m:oMath"),
             xml.count("<w:drawing"), os.path.getsize(DST) / 1048576))


if __name__ == "__main__":
    main()
