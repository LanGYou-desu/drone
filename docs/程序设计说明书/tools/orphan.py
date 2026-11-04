
import pymupdf
doc = pymupdf.open(r"D:\CodeWorks\drone\docs\程序设计说明书\main.pdf")
orphans = []
for i in range(doc.page_count):
    page = doc[i]
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    if not blocks: continue
    blocks.sort(key=lambda b: b["bbox"][3])
    last = blocks[-1]
    spans = [s for l in last["lines"] for s in l["spans"]]
    if not spans: continue
    txt = "".join(s["text"] for s in spans).strip()
    fonts = set(s["font"] for s in spans)
    if txt and all(("SimHei" in f) for f in fonts) and len(txt) < 40:
        orphans.append((i+1, txt))
print("orphan-heading candidates:", orphans)
