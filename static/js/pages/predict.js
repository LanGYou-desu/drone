/**
 * Predict Page — 轨迹预测 + 3D 预览
 * 高内聚：所有预测逻辑聚合在此模块
 * 低耦合：仅通过 API 与后端通信，不依赖其他页面模块
 */
import { toast } from '../common/toast.js';

// ---- 页面数据 ----
const { methodsData, predSettings } = window._PAGE_DATA_ || {};
const detectionMethods = methodsData || {};
const predCfg = predSettings || {};

// ---- 3D 场景 ----
let scene, camera, renderer, controls, THREE;
let lines = {}, predLines = {};

async function initViewer() {
    THREE = await import('three');
    const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');

    const container = document.getElementById('predictViewer');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0D0D0F);

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(10, 8, 12);
    camera.lookAt(4, 2, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(4, 2, 4);

    // 基础设施
    const grid = new THREE.GridHelper(200, 200, 0x2C2C2E, 0x1C1C1E);
    grid.position.y = -0.5;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(6));
    scene.add(new THREE.AmbientLight(0x445566, 0.8));
    const dir = new THREE.DirectionalLight(0xFFFFFF, 0.9);
    dir.position.set(6, 10, 4);
    scene.add(dir);

    function loop() {
        requestAnimationFrame(loop);
        controls.update();
        renderer.render(scene, camera);
    }
    loop();

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    drawTrails();
}

function drawTrails() {
    for (const id in lines) { removeObj(lines[id]); }
    for (const id in predLines) { removeObj(predLines[id]); }
    lines = {}; predLines = {};

    for (const [id, data] of Object.entries(detectionMethods)) {
        if (!data.points || data.points.length < 2) continue;
        const curve = new THREE.CatmullRomCurve3(data.points.map(p => new THREE.Vector3(p[0],p[1],p[2])));
        const pts = curve.getPoints(100);
        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: data.color })
        );
        scene.add(line);

        // 点球体
        const grp = new THREE.Group();
        const mat = new THREE.MeshStandardMaterial({ color: data.color, emissive: 0x111111 });
        data.points.forEach(p => {
            const s = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), mat);
            s.position.set(p[0], p[1], p[2]);
            grp.add(s);
        });
        scene.add(grp);
        lines[id] = { line, points: grp };
    }
}

function removeObj(obj) {
    if (!obj) return;
    if (obj.line) scene.remove(obj.line);
    if (obj.points) scene.remove(obj.points);
}

async function runPrediction() {
    const platform = document.getElementById('predictPlatform').value;
    const numPoints = parseInt(document.getElementById('numPointsSlider').value);
    const timeStep = parseFloat(document.getElementById('timeStep').value) || predCfg.defaultTimeStep;

    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.textContent = '⏳ 预测中...';

    try {
        const isAll = platform === 'all';
        const url = isAll ? '/api/predict_all' : '/api/predict';
        const body = isAll
            ? { num_points: numPoints, time_step: timeStep }
            : { method_id: platform, points: detectionMethods[platform]?.points || [],
                timestamps: detectionMethods[platform]?.timestamps || [],
                num_points: numPoints, time_step: timeStep };

        const resp = await fetch(url, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
        const result = await resp.json();

        // 清除旧预测线
        for (const id in predLines) { removeObj(predLines[id]); }
        predLines = {};

        if (result.success) {
            const results = isAll ? result.results : { [platform]: { prediction: result.prediction, pred_times: result.pred_times } };

            const container = document.getElementById('predictResults');
            let html = '';

            for (const [mid, pred] of Object.entries(results)) {
                if (!pred.prediction?.length) continue;
                // 渲染虚线
                const color = detectionMethods[mid]?.color || '#ffffff';
                const pts = pred.prediction.map(p => new THREE.Vector3(p[0],p[1],p[2]));
                const line = new THREE.Line(
                    new THREE.BufferGeometry().setFromPoints(pts),
                    new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.2 })
                );
                line.computeLineDistances();
                scene.add(line);

                const grp = new THREE.Group();
                const mat = new THREE.MeshStandardMaterial({ color, emissive: 0x111111, transparent: true, opacity: 0.6 });
                pred.prediction.forEach(p => {
                    const s = new THREE.Mesh(new THREE.SphereGeometry(0.04, 6, 6), mat);
                    s.position.set(p[0], p[1], p[2]);
                    grp.add(s);
                });
                scene.add(grp);
                predLines[mid] = { line, points: grp };

                // 结果文本
                html += `<div style="margin-bottom:8px;padding:8px 12px;background:var(--bg-input);border-radius:var(--radius-md);">
                    <span class="legend-dot" style="background:${color};color:${color};"></span>
                    <strong>${detectionMethods[mid]?.name || mid}</strong>
                    — ${pred.prediction.length} 个预测点
                    <br><span style="font-size:0.75rem;color:var(--text-secondary);">
                    终点: (${pred.prediction[pred.prediction.length-1].map(v=>v.toFixed(1)).join(', ')})
                    </span>
                </div>`;
            }
            container.innerHTML = html || '<p style="color:var(--text-secondary);">无有效预测结果</p>';
            toast.success('预测完成');
        } else {
            toast.error('预测失败: ' + (result.error || ''));
        }
    } catch (e) {
        toast.error('请求失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '⚡ 开始预测';
    }
}

// ---- 初始化 ----
document.addEventListener('DOMContentLoaded', () => {
    initViewer();
    document.getElementById('numPointsSlider').addEventListener('input', e => {
        document.getElementById('numPointsLabel').textContent = e.target.value;
    });
    document.getElementById('predictBtn').addEventListener('click', runPrediction);
});
