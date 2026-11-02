"""把数据和文字填进报告模板，生成 report.html。"""
import json

data = open("report_data.json", encoding="utf-8").read()
notes = json.dumps(json.load(open("notes.json", encoding="utf-8")), ensure_ascii=False)
d = json.loads(data)
# 体色色相（HSL）大致换算成 OKLCH 色相，用作页面强调色
hue = round((d["hue_mean"] + 5) % 360)
html = open("report_template.html", encoding="utf-8").read()
html = (html.replace("__DATA__", data.replace("</", "<\\/"))
            .replace("__NOTES__", notes.replace("</", "<\\/"))
            .replace("__HUE__", str(hue)))
open("report.html", "w", encoding="utf-8").write(html)
print("report.html", round(len(html.encode()) / 1024), "KB, hue", hue)
