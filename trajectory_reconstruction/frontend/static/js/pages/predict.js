/**
 * Predict Page — 轨迹预测 + 3D 预览 + 动画播放
 */
import { toast } from '../common/toast.js';
import { lerp } from '../common/utils.js';

const { methodsData, predSettings } = window._PAGE_DATA_ || {};
const detectionMethods = methodsData || {};
const predCfg = predSettings || {};

let scene, camera, renderer, controls, THREE;
let lines = {}, predLines = {};
let movingSpheres = {};
let animActive = false, animId = null, animSpeed = 1.0;
let animElapsed = 0;
let animRange = { start: 0, end: 0 };
let predictedData = {};

// ═══════════════════════════ 3D 场景 ═══════════════════════════

async function initViewer() {
    THREE = await import('three');
    const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');

    const container = document.getElementById('predictViewer');
    const W = container.clientWidth, H = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0d18);
    scene.fog = new THREE.Fog(0x0d0d18, 20, 100);

    camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 500);
    camera.position.set(10, 7, 12);
    camera.lookAt(4, 2, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(4, 2, 4);
    controls.autoRotate = false;

    // 基础设施
    scene.add(new THREE.GridHelper(100, 20, 0x334466, 0x1a1a2e).translateY(-0.5));
    scene.add(new THREE.AxesHelper(8));
    scene.add(new THREE.AmbientLight(0x445566, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(8, 12, 6);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x4466aa, 0.3);
    fill.position.set(-4, 2, -4);
    scene.add(fill);

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
        const curve = new THREE.CatmullRomCurve3(data.points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
        const pts = curve.getPoints(100);
        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: data.color, transparent: true, opacity: 0.85 })
        );
        scene.add(line);

        const grp = new THREE.Group();
        const mat = new THREE.MeshStandardMaterial({ color: data.color, emissive: data.color, emissiveIntensity: 0.3, roughness: 0.3 });
        data.points.forEach(p => {
            const s = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 10), mat);
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

// ═══════════════════════════ 预测 ═══════════════════════════

async function runPrediction() {
    const platform = document.getElementById('predictPlatform').value;
    const numPoints = parseInt(document.getElementById('numPointsSlider').value);
    const timeStep = parseFloat(document.getElementById('timeStep').value) || predCfg.defaultTimeStep || 0.5;
    const btn = document.getElementById('predictBtn');

    btn.disabled = true; btn.textContent = '预测中...';

    try {
        const isAll = platform === 'all';
        const body = isAll
            ? { num_points: numPoints, time_step: timeStep }
            : { method_id: platform, points: detectionMethods[platform]?.points || [],
                timestamps: detectionMethods[platform]?.timestamps || [],
                num_points: numPoints, time_step: timeStep };

        const resp = await fetch(isAll ? '/api/predict_all' : '/api/predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const result = await resp.json();

        // 清除旧预测线
        for (const id in predLines) { removeObj(predLines[id]); }
        predLines = {};
        predictedData = {};

        if (result.success) {
            const results = isAll ? result.results : { [platform]: { prediction: result.prediction, pred_times: result.pred_times } };
            const container = document.getElementById('predictResults');
            let html = '';

            for (const [mid, pred] of Object.entries(results)) {
                if (!pred.prediction?.length) continue;
                predictedData[mid] = pred;

                const color = detectionMethods[mid]?.color || '#ffffff';
                // 预测虚线：从原始轨迹最后一点连到预测点，消除断连
                const pts = [];
                const histPts = detectionMethods[mid]?.points;
                if (histPts?.length) {
                    const last = histPts[histPts.length - 1];
                    pts.push(new THREE.Vector3(last[0], last[1], last[2]));
                }
                pred.prediction.forEach(p => pts.push(new THREE.Vector3(p[0], p[1], p[2])));
                const line = new THREE.Line(
                    new THREE.BufferGeometry().setFromPoints(pts),
                    new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.5 })
                );
                line.computeLineDistances();
                scene.add(line);

                // 预测点球体
                const grp = new THREE.Group();
                const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.2, transparent: true, opacity: 0.5 });
                pred.prediction.forEach(p => {
                    const s = new THREE.Mesh(new THREE.SphereGeometry(0.04, 8, 8), mat);
                    s.position.set(p[0], p[1], p[2]);
                    grp.add(s);
                });
                scene.add(grp);
                predLines[mid] = { line, points: grp };

                html += `<div style="margin-bottom:6px;padding:8px 12px;background:var(--bg-input);border-radius:var(--radius-md);">
                    <span class="legend-dot" style="background:${color};color:${color};"></span>
                    <strong>${detectionMethods[mid]?.name || mid}</strong>
                    — ${pred.prediction.length} 预测点
                </div>`;
            }
            container.innerHTML = html || '<p style="color:var(--text-secondary);">无有效预测结果</p>';
            toast.success('预测完成，点击 ▶ 播放动画');
        } else {
            toast.error('预测失败: ' + (result.error || ''));
        }
    } catch (e) {
        toast.error('请求失败: ' + e.message);
    } finally {
        btn.disabled = false; btn.textContent = '⚡ 开始预测';
    }
}

// ═══════════════════════════ 动画播放（历史+预测全程）══════════════════════════


function calcRange() {
    let min = Infinity, max = -Infinity;
    // 历史轨迹时间范围
    for (const id in detectionMethods) {
        const ts = detectionMethods[id]?.timestamps;
        if (ts?.length) { min = Math.min(min, ts[0]); max = Math.max(max, ts[ts.length-1]); }
    }
    // 预测时间范围
    for (const id in predictedData) {
        const times = predictedData[id]?.pred_times;
        if (times?.length) max = Math.max(max, times[times.length-1]);
    }
    animRange = { start: min === Infinity ? 0 : min, end: max === -Infinity ? 0 : max };
}

function makeGlowSphere(color) {
    const g = new THREE.Group();
    g.add(new THREE.Mesh(
        new THREE.SphereGeometry(0.15, 16, 16),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.8, roughness: 0.2 })));
    g.add(new THREE.Mesh(
        new THREE.SphereGeometry(0.3, 12, 12),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.2, blending: THREE.AdditiveBlending, depthWrite: false })));
    return g;
}

function startAnim(fromStart = false) {
    if (fromStart) animElapsed = 0;
    if (Object.keys(predictedData).length === 0) { toast.warning('请先进行预测'); return; }
    if (!THREE) { toast.warning('3D 场景加载中，请稍候'); return; }

    calcRange();
    if (animRange.end <= animRange.start) { toast.warning('无有效时间数据'); return; }
    if (animElapsed >= animRange.end - animRange.start) animElapsed = 0;

    animActive = true;

    // 收集参与播放的平台
    const playIds = new Set();
    for (const id in predictedData) playIds.add(id);
    for (const [id, d] of Object.entries(detectionMethods)) {
        if (d.visible && d.points?.length >= 2) playIds.add(id);
    }

    // 仅在首次播放或重播时创建球体
    if (fromStart || Object.keys(movingSpheres).length === 0) {
        for (const id in movingSpheres) { scene.remove(movingSpheres[id]); delete movingSpheres[id]; }
        for (const id of playIds) {
            const color = detectionMethods[id]?.color || '#fff';
            const sphere = makeGlowSphere(color);
            scene.add(sphere);
            movingSpheres[id] = sphere;
        }
    }

    const startWall = performance.now();
    const startElapsed = animElapsed;

    const step = now => {
        if (!animActive) {
            animElapsed = startElapsed + (now - startWall) / 1000 * animSpeed;
            return;
        }
        animElapsed = startElapsed + (now - startWall) / 1000 * animSpeed;
        const ts = animRange.start + animElapsed;

        if (ts >= animRange.end) {
            for (const id of playIds) {
                const pred = predictedData[id];
                if (pred?.prediction?.length) {
                    const last = pred.prediction[pred.prediction.length-1];
                    movingSpheres[id]?.position.set(last[0], last[1], last[2]);
                }
            }
            pauseAnim();
            return;
        }

        for (const id of playIds) {
            const data = detectionMethods[id];
            const s = movingSpheres[id];
            if (!s) continue;
            const histPts = data?.points, histTs = data?.timestamps;
            if (histPts?.length && histTs?.length && ts <= histTs[histTs.length-1]) {
                const pos = lerp(histPts, histTs, ts);
                if (pos) { s.position.set(pos[0], pos[1], pos[2]); s.visible = true; continue; }
            }
            const pred = predictedData[id];
            if (pred?.prediction?.length && pred?.pred_times?.length && ts <= pred.pred_times[pred.pred_times.length-1]) {
                const pos = lerp(pred.prediction, pred.pred_times, ts);
                if (pos) { s.position.set(pos[0], pos[1], pos[2]); s.visible = true; continue; }
            }
            s.visible = false;
        }
        animId = requestAnimationFrame(step);
    };
    animId = requestAnimationFrame(step);
    toast.show('全程轨迹播放');
}

function pauseAnim() {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    // 球体保持在当前位置不消失
}

function stopAnim() {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    animElapsed = 0;
    for (const id in movingSpheres) {
        scene.remove(movingSpheres[id]);
        delete movingSpheres[id];
    }
}

// ═══════════════════════════ 初始化 ═══════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initViewer();

    document.getElementById('numPointsSlider').addEventListener('input', e => {
        document.getElementById('numPointsLabel').textContent = e.target.value;
    });

    document.getElementById('predictBtn').addEventListener('click', runPrediction);

    // 播放/停止
    document.getElementById('playBtn').addEventListener('click', () => startAnim(false));
    document.getElementById('stopBtn').addEventListener('click', pauseAnim);

    // 倍速
    document.getElementById('speedSelect').addEventListener('change', e => {
        animSpeed = parseFloat(e.target.value);
    });
});
