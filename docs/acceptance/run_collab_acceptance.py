# -*- coding: utf-8 -*-
"""集成验收：总览综合重建 / 预测联动 / 分析评估"""
import json, os, time
from pathlib import Path
import numpy as np
import requests
from playwright.sync_api import sync_playwright

ROOT = Path(r'D:\CodeWorks\drone')
BASE = 'http://127.0.0.1:5000'
OUT = ROOT / 'docs' / 'acceptance'
report = {'pages': {}, 'synthesis': {}, 'prediction': {}, 'assessment': {}, 'ui': {}, 'errors': []}

print('== 1) 页面 ==')
for path in ('/', '/predict', '/analysis/', '/ai', '/data', '/settings', '/docs'):
    r = requests.get(BASE + path, timeout=30)
    report['pages'][path] = r.status_code
    print('  %-12s %s' % (path, r.status_code))

print('\n== 2) 综合重建（总览链路）==')
t0 = time.time()
syn = requests.post(BASE + '/api/synthesize', timeout=120).json()
report['synthesis'] = {
    'mode': syn.get('mode'), 'points': syn.get('point_count'),
    'selected': (syn.get('fusion') or {}).get('selected'),
    'excluded': (syn.get('fusion') or {}).get('excluded'),
    'agreement': (syn.get('fusion') or {}).get('agreement'),
    'mean_sigma': (syn.get('reconstruction') or {}).get('mean_sigma'),
    'elapsed_ms': round((time.time() - t0) * 1000),
}
print('  mode=%s points=%s selected=%s agreement=%s sigma=%s (%.0f ms)' % (
    syn.get('mode'), syn.get('point_count'), report['synthesis']['selected'],
    report['synthesis']['agreement'], report['synthesis']['mean_sigma'], report['synthesis']['elapsed_ms']))

print('\n== 3) 预测联动（综合轨迹）==')
pr = requests.post(BASE + '/api/predict', json={'method_id': 'synthetic', 'num_points': 10, 'time_step': 0.2},
                   timeout=120).json()
pts = np.array(pr['prediction'], dtype=float)
steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
chord = float(np.linalg.norm(pts[-1] - pts[0]))
u = (pts[-1] - pts[0]) / max(chord, 1e-9)
d = pts - pts[0]
bend = float(np.linalg.norm(d - np.outer(d @ u, u), axis=1).max() / max(chord, 1e-9))
report['prediction'] = {
    'points': len(pts), 'times': [pr['pred_times'][0], pr['pred_times'][-1]],
    'bend': round(bend, 4),
    'step_min': round(float(steps.min()), 3), 'step_max': round(float(steps.max()), 3),
    'first_equals_last_history': bool(np.allclose(pts[0], pts[0])),
}
print('  点数=%d 时间=%.1f~%.1f 弯曲度=%.3f 步长 %.2f~%.2f m' % (
    len(pts), pr['pred_times'][0], pr['pred_times'][-1], bend, steps.min(), steps.max()))

print('\n== 4) 智能评估 ==')
t0 = time.time()
ass = requests.post(BASE + '/api/sensors/assess', json={'scenario': 'live'}, timeout=120).json()
report['assessment'] = {
    'score': ass.get('score'), 'level': ass.get('level'),
    'prediction_source': (ass.get('prediction') or {}).get('source'),
    'factors': [(f['label'], round(f['score'], 2)) for f in ass.get('factors', [])],
    'recommendations': [r['action'] for r in ass.get('recommendations', [])],
    'elapsed_ms': round((time.time() - t0) * 1000),
}
print('  %s %.1f 分 | 预测来源 %s | 建议 %s' % (
    ass.get('level'), ass.get('score'), report['assessment']['prediction_source'],
    report['assessment']['recommendations'][:2]))

print('\n== 5) 浏览器 ==')
args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-sandbox', '--no-sandbox']
with sync_playwright() as p:
    browser = None
    for kw in ({'executable_path': str(Path(os.environ['LOCALAPPDATA']) / 'ms-playwright' /
                                      'chromium-1228' / 'chrome-win64' / 'chrome.exe')}, {'channel': 'msedge'}):
        try:
            browser = p.chromium.launch(headless=True, args=args, **kw); break
        except Exception as exc:
            print('launch failed', str(exc)[:80])
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    errors = []
    page.on('console', lambda m: errors.append('console: ' + m.text) if m.type == 'error' else None)
    page.on('pageerror', lambda e: errors.append('pageerror: ' + str(e)))

    page.goto(BASE + '/', wait_until='networkidle', timeout=60000)
    page.wait_for_timeout(3500)
    page.screenshot(path=str(OUT / 'dashboard_recon.png'))
    report['ui']['dashboard'] = {
        'recon': (page.text_content('#reconSummary') or '').replace('\n', ' ')[:200],
        'canvas': page.evaluate("() => { const c = document.querySelector('#viewer canvas'); return c ? [c.width, c.height] : null; }"),
    }

    page.goto(BASE + '/predict', wait_until='networkidle', timeout=60000)
    page.wait_for_timeout(2000)
    platform = page.input_value('#predictPlatform')
    with page.expect_response(lambda r: '/api/predict' in r.url, timeout=60000) as info:
        page.click('#predictBtn')
    page.wait_for_timeout(2500)
    page.screenshot(path=str(OUT / 'predict_linked.png'))
    report['ui']['predict'] = {
        'default_platform': platform,
        'recon_status': (page.text_content('#reconStatus') or '').strip()[:160],
        'horizon_hint': (page.text_content('#horizonHint') or '').strip(),
        'pred_label': (page.text_content('#pred-synthetic') or '').strip(),
        'api_status': info.value.status,
    }

    page.goto(BASE + '/analysis/', wait_until='networkidle', timeout=60000)
    page.wait_for_timeout(4000)
    page.screenshot(path=str(OUT / 'analysis_assessment.png'), full_page=True)
    report['ui']['analysis'] = {
        'risk_level': (page.text_content('#riskLevel') or '').strip(),
        'status': (page.text_content('#assessStatus') or '').strip(),
        'charts': page.evaluate("() => document.querySelectorAll('.chart canvas').length"),
        'gauge': page.evaluate("() => document.querySelectorAll('#riskGauge canvas').length"),
        'timeline': page.evaluate("() => document.querySelectorAll('#riskTimeline canvas').length"),
    }
    report['errors'] = errors
    print('  总览:', report['ui']['dashboard']['recon'][:80])
    print('  预测:', report['ui']['predict']['default_platform'], '|',
          report['ui']['predict']['horizon_hint'], '|', report['ui']['predict']['pred_label'])
    print('  分析评估:', report['ui']['analysis']['risk_level'], '| charts',
          report['ui']['analysis']['charts'], '| gauge', report['ui']['analysis']['gauge'])
    print('  控制台错误:', len(errors))
    browser.close()

(OUT / 'collab_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n报告:', OUT / 'collab_report.json')
