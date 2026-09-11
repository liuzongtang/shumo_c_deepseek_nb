# -*- coding: utf-8 -*-
"""生成中文版 pandoc reference.docx：正文宋体、标题黑体（拉丁/数字用 Times New Roman）。
用法：python _make_reference.py  ->  产出 reference_cn.docx，供
      pandoc 论文初稿.md --reference-doc=reference_cn.docx -o 论文初稿.docx
"""
import subprocess, zipfile, re


def main():
    data = subprocess.check_output(["pandoc", "--print-default-data-file", "reference.docx"])
    with open("reference_cn.docx", "wb") as f:
        f.write(data)

    zin = zipfile.ZipFile("reference_cn.docx")
    names = zin.namelist()
    content = {n: zin.read(n) for n in names}
    zin.close()

    xml = content["word/styles.xml"].decode("utf-8")
    n_heading = len(re.findall(r'<w:rFonts[^>]*majorEastAsia[^>]*/>', xml))
    n_body = len(re.findall(r'<w:rFonts[^>]*minorEastAsia[^>]*/>', xml))
    xml = re.sub(r'<w:rFonts[^>]*majorEastAsia[^>]*/>',
                 '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体" />', xml)
    xml = re.sub(r'<w:rFonts[^>]*minorEastAsia[^>]*/>',
                 '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" />', xml)
    content["word/styles.xml"] = xml.encode("utf-8")

    with zipfile.ZipFile("reference_cn.docx", "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, content[n])

    print("reference_cn.docx 已生成：标题黑体(%d处)、正文宋体(%d处)" % (n_heading, n_body))


if __name__ == "__main__":
    main()
