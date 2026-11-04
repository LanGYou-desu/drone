
import pymupdf
doc = pymupdf.open(r"D:\CodeWorks\drone\docs\程序设计说明书\main.pdf")
blank = []
for i in range(doc.page_count):
    page = doc[i]
    t = page.get_text().strip()
    imgs = page.get_images(full=True)
    drawings = page.get_drawings()
    if len(t) < 120 and not imgs and len(drawings) < 5:
        blank.append((i+1, len(t), len(imgs), len(drawings)))
print("pages", doc.page_count)
print("near-blank:", blank)
