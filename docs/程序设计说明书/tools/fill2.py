
import pymupdf
doc = pymupdf.open(r"D:\CodeWorks\drone\docs\程序设计说明书\main.pdf")
res = []
for i in range(doc.page_count):
    page = doc[i]
    H = page.rect.height
    top, bottom = 70, H - 55
    ys = []
    for b in page.get_text("blocks"):
        if b[1] > top and b[3] < bottom: ys.append(b[3])
    for d in page.get_drawings():
        r = d["rect"]
        if r.y0 > top and r.y1 < bottom: ys.append(r.y1)
    for img in page.get_image_info():
        bb = img["bbox"]
        if bb[1] > top and bb[3] < bottom: ys.append(bb[3])
    if not ys: 
        res.append((0.0, i+1)); continue
    fill = (max(ys) - top) / (bottom - top)
    res.append((fill, i+1))
res.sort()
print("pages", doc.page_count)
print("lowest:", [(p, round(f,2)) for f,p in res[:15]])
