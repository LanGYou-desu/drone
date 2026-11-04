
import os, traceback
from playwright.sync_api import sync_playwright

OUT = r"D:\CodeWorks\drone\docs\程序设计说明书\figures\ui"
os.makedirs(OUT, exist_ok=True)
log = []

PAGES = [
    ("dashboard",        "http://127.0.0.1:5000/",        "wait", 5000),
    ("predict",          "http://127.0.0.1:5000/predict", "click:#predictBtn", 6000),
    ("analysis",         "http://127.0.0.1:5000/analysis/","wait", 6000),
    ("ai",               "http://127.0.0.1:5000/ai",      "wait", 2000),
    ("data",             "http://127.0.0.1:5000/data",    "wait", 3000),
    ("settings",         "http://127.0.0.1:5000/settings","wait", 2500),
    ("docs",             "http://127.0.0.1:5000/docs",    "wait", 2000),
    ("detect",           "http://127.0.0.1:5001/",        "wait", 3000),
    ("detect_settings",  "http://127.0.0.1:5001/settings","wait", 2500),
    ("detect_history",   "http://127.0.0.1:5001/history", "wait", 3000),
]

def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True,
            args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--disable-gpu-sandbox"])
        ctx = browser.new_context(viewport={"width":1600,"height":1000}, device_scale_factor=2)
        ctx.add_init_script("try{localStorage.setItem('theme','dark');}catch(e){}")
        for name, url, action, wait in PAGES:
            page = ctx.new_page()
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)[:300]))
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                page.wait_for_timeout(1500)
                if action.startswith("click:"):
                    sel = action.split(":",1)[1]
                    try:
                        page.click(sel, timeout=10000)
                        log.append("clicked %s on %s" % (sel, name))
                    except Exception as e:
                        log.append("CLICK FAIL %s on %s: %s" % (sel, name, str(e)[:200]))
                page.wait_for_timeout(wait)
                p = os.path.join(OUT, name + ".png")
                page.screenshot(path=p)
                h = page.evaluate("document.documentElement.scrollHeight")
                log.append("saved %-16s %7d bytes  scrollH=%s  errors=%s" % (name, os.path.getsize(p), h, errs[:3]))
            except Exception as e:
                log.append("FAIL %s: %s" % (name, str(e)[:300]))
            finally:
                page.close()
        browser.close()

try:
    main()
except Exception:
    log.append(traceback.format_exc())
open(r"D:\CodeWorks\drone\docs\程序设计说明书\tools\shot_log.txt","w",encoding="utf-8").write("\n".join(log))
