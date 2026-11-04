
from PIL import Image
base = r"D:\CodeWorks\drone\docs\程序设计说明书\figures\ui"
crops = {"ai.png":0.62, "data.png":0.68, "detect_history.png":0.80}
for name, frac in crops.items():
    p = base + "\\" + name
    im = Image.open(p)
    w,h = im.size
    im.crop((0,0,w,int(h*frac))).save(p)
    print(name, im.size, "->", Image.open(p).size)
