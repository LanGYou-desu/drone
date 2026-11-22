# -*- coding: utf-8 -*-
"""自动调整视角直到预测终点标签完整进入画面, 然后截图"""
import os
from pathlib import Path
import numpy as np
import requests
from playwright.sync_api import sync_playwright

ROOT = Path(r'D:\CodeWorks\drone')
BASE = 'http://127.0.0.1:5000'
OUT = ROOT / 'docs' / 'acceptance'

for mid in ('visible', 'infrared', 'radar'):
    requests.post(BASE + '/api/toggle_method', json={'method_id': mid}, timeout=30)
src = np.loadtxt(ROOT / 'data' / 'fact' / 'visible.dat')
np.savetxt(ROOT / 'data' / '_accept_history.dat', src[:54], fmt='%.4f %.4f %.4f %.3f')
with open(ROOT / 'data' / '_accept_history.dat', 'rb') as f:
    requests.post(BASE + '/api/load_data', files={'file': ('demo_history.dat', f, 'text/plain')},
                  data={'method_id': 'self'}, timeout=60)

LABEL_RECT = """() => {
  const els = Array.from(document.querySelectorAll('#predictViewer div'));
  const el = els.find(e => e.children.length === 0 && e.textContent.includes('预测终点'));
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, text: el.textContent.trim()};
}"""

args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-sandbox', '--no-sandbox']
with sync_playwright() as p:
    browser = None
    for kw in ({'executable_path': str(Path(os.environ['LOCALAPPDATA']) / 'ms-playwright' /
                                      'chromium-1228' / 'chrome-win64' / 'chrome.exe')}, {'channel': 'msedge'}):
        try:
            browser = p.chromium.launch(headless=True, args=args, **kw); break
        except Exception as e:
            print('launch failed', str(e)[:80])
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    page.goto(BASE + '/predict', wait_until='networkidle', timeout=60000)
    page.evaluate("() => { try { sessionStorage.clear(); } catch (e) {} }")
    page.reload(wait_until='networkidle')
    page.wait_for_selector('#predictPlatform', timeout=30000)
    page.select_option('#predictPlatform', 'self')
    page.fill('#timeStep', '0.2')
    page.eval_on_selector('#numPointsSlider', "el => { el.value = 10; el.dispatchEvent(new Event('input')); }")
    with page.expect_response(lambda r: '/api/predict' in r.url, timeout=60000) as info:
        page.click('#predictBtn')
    page.wait_for_timeout(2200)

    page.mouse.move(720, 450)
    for step in range(14):
        rect = page.evaluate(LABEL_RECT)
        ok = rect and 20 < rect['left'] and rect['right'] < 1420 and 80 < rect['top'] and rect['bottom'] < 820
        print('wheel step %d -> label rect %s' % (step, rect))
        if ok:
            break
        page.mouse.wheel(0, 200)
        page.wait_for_timeout(250)
    page.wait_for_timeout(1200)
    page.screenshot(path=str(OUT / 'kinflow_predict_overview.png'))
    print('saved overview')
    browser.close()

for mid in ('visible', 'infrared', 'radar'):
    requests.post(BASE + '/api/toggle_method', json={'method_id': mid}, timeout=30)
print('restored')
