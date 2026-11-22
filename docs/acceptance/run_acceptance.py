# -*- coding: utf-8 -*-
"""端到端验收: 网页上完成预测并核对坐标 + 截图 + 与留出真值比较"""
import glob, json, os
from pathlib import Path

import numpy as np
import requests
from playwright.sync_api import sync_playwright

ROOT = Path(r'D:\CodeWorks\drone')
BASE = 'http://127.0.0.1:5000'
OUT = ROOT / 'docs' / 'acceptance'
OUT.mkdir(parents=True, exist_ok=True)


def upload_history(hist, name):
    path = ROOT / 'data' / '_accept_history.dat'
    np.savetxt(path, hist, fmt='%.4f %.4f %.4f %.3f')
    with open(path, 'rb') as f:
        resp = requests.post(BASE + '/api/load_data',
                             files={'file': (name, f, 'text/plain')},
                             data={'method_id': 'self'}, timeout=60)
    return resp.status_code, resp.json()


def pick_test_trajectory():
    for f in sorted(glob.glob(str(ROOT / 'train' / 'kinflow_predictor' / 'dataset' / 'test' / '*.npz'))):
        d = np.load(f)
        if len(d['positions']) >= 80:
            pos = d['positions'].astype(float)
            ts = d['timestamps'].astype(float)
            rows = np.column_stack([pos[:60], ts[:60]])
            return Path(f).name, rows[:50], rows[50:60]
    raise RuntimeError('no suitable test trajectory')


def launch(p):
    args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader',
            '--disable-gpu-sandbox', '--no-sandbox']
    for kw in (
        {'executable_path': str(Path(os.environ['LOCALAPPDATA']) / 'ms-playwright' /
                                'chromium-1228' / 'chrome-win64' / 'chrome.exe')},
        {'channel': 'msedge'},
    ):
        try:
            return p.chromium.launch(headless=True, args=args, **kw)
        except Exception as exc:
            print('launch failed:', str(exc).splitlines()[0][:100])
    raise RuntimeError('no usable browser')


def ui_predict(page, num_points, time_step, screenshot):
    page.goto(BASE + '/predict', wait_until='networkidle', timeout=60000)
    page.evaluate("() => { try { sessionStorage.clear(); } catch (e) {} }")
    page.reload(wait_until='networkidle', timeout=60000)
    page.wait_for_selector('#predictPlatform', timeout=30000)
    page.select_option('#predictPlatform', 'self')
    page.fill('#timeStep', str(time_step))
    page.eval_on_selector('#numPointsSlider',
                          f"el => {{ el.value = {num_points}; el.dispatchEvent(new Event('input')); }}")
    with page.expect_response(lambda r: '/api/predict' in r.url, timeout=60000) as info:
        page.click('#predictBtn')
    api = info.value.json()
    page.wait_for_timeout(2500)
    shot = OUT / screenshot
    page.screenshot(path=str(shot))
    predicted = json.loads(page.evaluate("() => sessionStorage.getItem('predictedData') || '{}'"))
    return {
        'status': info.value.status,
        'api': api,
        'label': page.text_content('#pred-self'),
        'slider_points': page.text_content('#numPointsLabel'),
        'time_step_input': page.input_value('#timeStep'),
        'session_keys': list(predicted.keys()),
        'rendered_points': len(predicted.get('self', {}).get('prediction', [])),
        'canvas': page.evaluate("() => { const c = document.querySelector('#predictViewer canvas'); return c ? [c.width, c.height] : null; }"),
        'screenshot': str(shot),
    }


def compare(hist_rows, gt_rows, api, label):
    pred = np.array(api['prediction'], dtype=float)
    times = np.array(api['pred_times'], dtype=float)
    anchor = hist_rows[-1, :3]
    future, ft = pred[1:], times[1:]
    errs = []
    for p, t in zip(future, ft):
        if t <= gt_rows[-1, 3] + 1e-6:
            j = int(np.argmin(np.abs(gt_rows[:, 3] - t)))
            errs.append(float(np.linalg.norm(p - gt_rows[j, :3])))
    errs = np.array(errs) if errs else np.array([np.nan])

    v = (hist_rows[-1, :3] - hist_rows[-2, :3]) / (hist_rows[-1, 3] - hist_rows[-2, 3])
    lin_err = []
    for gt in gt_rows[:len(errs)]:
        lin = hist_rows[-1, :3] + v * (gt[3] - hist_rows[-1, 3])
        lin_err.append(float(np.linalg.norm(lin - gt[:3])))
    lin_err = np.array(lin_err)

    return {
        'case': label,
        'compared_steps': int(len(errs)),
        'per_step_error_m': [round(float(e), 4) for e in errs],
        'ade_m': round(float(np.nanmean(errs)), 4),
        'fde_m': round(float(errs[-1]), 4),
        'linear_ade_m': round(float(np.mean(lin_err)), 4),
        'anchor_matches_history_end': bool(np.allclose(pred[0], anchor, atol=1e-3)),
        'times_monotonic': bool(np.all(np.diff(times) > 0)),
        'all_points_3d': bool(pred.shape[1] == 3),
        'first_pred': [round(float(x), 4) for x in future[0]],
        'gt_first': [round(float(x), 4) for x in gt_rows[0, :3]],
    }


report = {'cases': []}
src_demo = np.loadtxt(ROOT / 'data' / 'fact' / 'visible.dat')
demo_hist, demo_gt = src_demo[:54], src_demo[54:60]
test_name, test_hist, test_gt = pick_test_trajectory()
print('demo history %d pts, held-out %d pts' % (len(demo_hist), len(demo_gt)))
print('in-distribution test trajectory:', test_name, test_hist.shape, test_gt.shape)

with sync_playwright() as p:
    browser = launch(p)
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    errors = []
    page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    page.on('pageerror', lambda e: errors.append('pageerror: ' + str(e)))

    # --- 场景 A: 运行时演示轨迹（域外） ---
    code, body = upload_history(demo_hist, 'demo_history.dat')
    print('upload demo ->', code, body)
    ui_a = ui_predict(page, 10, 0.2, 'kinflow_predict_demo.png')
    case_a = compare(demo_hist, demo_gt, ui_a['api'], 'runtime demo trajectory (visible.dat, held-out 1.2 s)')
    case_a.update({k: ui_a[k] for k in ('status', 'label', 'slider_points', 'rendered_points', 'canvas', 'screenshot')})
    report['cases'].append(case_a)

    # --- 场景 B: 同分布的测试集轨迹（严格留出） ---
    code, body = upload_history(test_hist, 'test_history.dat')
    print('upload test ->', code, body)
    ui_b = ui_predict(page, 10, 0.1, 'kinflow_predict_test_traj.png')
    case_b = compare(test_hist, test_gt, ui_b['api'], f'in-distribution test trajectory ({test_name}, held-out 1.0 s)')
    case_b.update({k: ui_b[k] for k in ('status', 'label', 'slider_points', 'rendered_points', 'canvas', 'screenshot')})
    report['cases'].append(case_b)

    # --- 全平台预测（综合轨迹） ---
    page.select_option('#predictPlatform', 'all')
    with page.expect_response(lambda r: '/api/predict_all' in r.url, timeout=90000) as info:
        page.click('#predictBtn')
    all_api = info.value.json()
    page.wait_for_timeout(2000)
    page.screenshot(path=str(OUT / 'kinflow_predict_all.png'))
    report['predict_all'] = {
        'status': info.value.status,
        'success': all_api.get('success'),
        'platforms': sorted((all_api.get('results') or {}).keys()),
        'screenshot': str(OUT / 'kinflow_predict_all.png'),
    }
    report['console_errors'] = errors
    browser.close()

with open(OUT / 'acceptance_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

for c in report['cases']:
    print('\n== %s ==' % c['case'])
    print('  HTTP %s | 页面标签: %s | 滑块点数: %s | 渲染点数: %s | canvas: %s' % (
        c['status'], c['label'], c['slider_points'], c['rendered_points'], c['canvas']))
    print('  首点=历史末点: %s | 时间单调: %s | 3D 点: %s' % (
        c['anchor_matches_history_end'], c['times_monotonic'], c['all_points_3d']))
    print('  每步误差(m):', c['per_step_error_m'])
    print('  KinFlow ADE %.4f / FDE %.4f  |  线性外推 ADE %.4f' % (c['ade_m'], c['fde_m'], c['linear_ade_m']))
    print('  截图:', c['screenshot'])
print('\n== 全平台 ==', report['predict_all'])
print('控制台错误:', report['console_errors'])
