/**
 * Predict Page — 轨迹预测 + 3D 预览 + 动画播放
 */
import { toast } from '../common/toast.js';
import { lerp } from '../common/utils.js';
import { buildAxes } from '../common/three-utils.js';

const { methodsData, predSettings } = window._PAGE_DATA_ || {};
const detectionMethods = methodsData || {};
const predCfg = predSettings || {};

let scene, camera, renderer, controls, THREE, CSS2DObject;
let lines = {}, predLines = {};
let movingSpheres = {};
let animActive = false, animId = null, animSpeed = 1.0;
let animElapsed = 0;
let animRange = { start: 0, end: 0 };
const PREDICT_KEY = 'predictedData';

// 从 sessionStorage 恢复预测数据
let predictedData = {};
try {
    const saved = sessionStorage.getItem(PREDICT_KEY);
    if (saved) predictedData = JSON.parse(saved);
} catch {}

function savePredictedData() {
    try { sessionStorage.setItem(PREDICT_KEY, JSON.stringify(predictedData)); } catch {}
}

// ═══════════════════════════ 3D 场景 ═══════════════════════════

async function initViewer() {
    THREE = await import('three');
    const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
    const css = await import('three/addons/renderers/CSS2DRenderer.js');
    CSS2DObject = css.CSS2DObject;

    const container = document.getElementById('predictViewer');
    const W = window.innerWidth, H = window.innerHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0d18);
    scene.fog = new THREE.Fog(0x0d0d18, 30, 120);

    camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 500);
    camera.position.set(12, 7, 14);
    camera.lookAt(4, 2, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    const labelRenderer = new css.CSS2DRenderer();
    labelRenderer.setSize(W, H);
    labelRenderer.domElement.style.cssText = 'position:absolute;top:0;pointer-events:none;';
    container.appendChild(labelRenderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(4, 2, 4);
    controls.autoRotate = false;

    // 地面（与总览一致）
    const gGeo = new THREE.PlaneGeometry(200, 200);
    const gMat = new THREE.MeshStandardMaterial({ color: 0x1a1a28, roughness: 0.95, metalness: 0.2, transparent: true, opacity: 0.6 });
    const plane = new THREE.Mesh(gGeo, gMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.55;
    plane.receiveShadow = true;
    scene.add(plane);

    // 网格
    const grid = new THREE.GridHelper(100, 20, 0x334466, 0x1a1a2e);
    grid.position.y = -0.5;
    scene.add(grid);

    buildAxes(scene, CSS2DObject);
    scene.add(new THREE.AmbientLight(0x334466, 0.8));
    scene.add(new THREE.HemisphereLight(0x8899cc, 0x223344, 0.5));
    const sun = new THREE.DirectionalLight(0xffeedd, 1.2);
    sun.position.set(12, 20, 8); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 100; sun.shadow.camera.left = -30; sun.shadow.camera.right = 30; sun.shadow.camera.top = 30; sun.shadow.camera.bottom = -30; sun.shadow.bias = -0.0001;
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x4466aa, 0.3);
    fill.position.set(-4, 2, -4);
    scene.add(fill);

    function loop() {
        requestAnimationFrame(loop);
        controls.update();
        renderer.render(scene, camera);
        labelRenderer.render(scene, camera);
    }
    loop();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
    });

    drawTrails();
    renderCachedPredictions();
    calcRange();
}

function renderCachedPredictions() {
    if (!Object.keys(predictedData).length) return;
    for (const [mid, pred] of Object.entries(predictedData)) {
        if (!pred.prediction?.length) continue;
        const color = detectionMethods[mid]?.color || '#ffffff';
        const pts = pred.prediction.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.5 })
        );
        line.computeLineDistances();
        scene.add(line);
        const grp = new THREE.Group();
        const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.2, transparent: true, opacity: 0.5 });
        pred.prediction.forEach(p => {
            const s = new THREE.Mesh(new THREE.SphereGeometry(0.04, 8, 8), mat);
            s.position.set(p[0], p[1], p[2]);
            grp.add(s);
        });
        scene.add(grp);
        predLines[mid] = { line, points: grp };
    }
    // 在平台状态面板下显示预测结果
    for (const [mid, pred] of Object.entries(predictedData)) {
        const el = document.getElementById('pred-' + mid);
        if (el) {
            el.style.display = '';
            el.textContent = '预测 ' + (pred.prediction.length - 1) + ' 点';
        }
    }
}

// 起点标记
let startMarkers = {};

function drawTrails() {
    for (const id in lines) { removeObj(lines[id]); }
    for (const id in predLines) { removeObj(predLines[id]); }
    for (const id in startMarkers) { scene.remove(startMarkers[id]); delete startMarkers[id]; }
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

        // 起点标签
        const div = document.createElement('div');
        div.textContent = '🚀 起点';
        div.style.cssText = 'color:#fff;font-size:10px;font-weight:600;background:rgba(0,0,0,0.7);padding:2px 8px;border-radius:10px;';
        const label = new CSS2DObject(div);
        label.position.set(data.points[0][0], data.points[0][1], data.points[0][2]);
        scene.add(label);
        startMarkers[id] = label;
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

            for (const [mid, pred] of Object.entries(results)) {
                if (!pred.prediction?.length) continue;
                predictedData[mid] = pred;

                const color = detectionMethods[mid]?.color || '#ffffff';
                const pts = pred.prediction.map(p => new THREE.Vector3(p[0], p[1], p[2]));
                const line = new THREE.Line(
                    new THREE.BufferGeometry().setFromPoints(pts),
                    new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.5 })
                );
                line.computeLineDistances();
                scene.add(line);

                const grp = new THREE.Group();
                const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.2, transparent: true, opacity: 0.5 });
                pred.prediction.forEach(p => {
                    const s = new THREE.Mesh(new THREE.SphereGeometry(0.04, 8, 8), mat);
                    s.position.set(p[0], p[1], p[2]);
                    grp.add(s);
                });
                scene.add(grp);
                predLines[mid] = { line, points: grp };

                // 在平台状态下方显示预测点数
                const el = document.getElementById('pred-' + mid);
                if (el) { el.style.display = ''; el.textContent = '预测 ' + (pred.prediction.length - 1) + ' 点'; }
            }
            savePredictedData();
            calcRange();
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
    document.getElementById('animTimeLabel').textContent = animRange.start.toFixed(1) + 's';
    document.getElementById('animEndLabel').textContent = animRange.end.toFixed(1) + 's';
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

function togglePlay() {
    if (animActive) { pauseAnim(); return; }
    startAnim(animElapsed >= animRange.end - animRange.start);
}

function startAnim(fromStart = false) {
    if (fromStart) animElapsed = 0;
    if (!THREE) { toast.warning('3D 场景加载中，请稍候'); return; }

    calcRange();
    if (animRange.end <= animRange.start) { toast.warning('无有效时间数据'); return; }
    if (animElapsed >= animRange.end - animRange.start) animElapsed = 0;

    animActive = true;
    document.getElementById('playBtn').innerHTML = '⏸ 暂停';

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
    const duration = animRange.end - animRange.start;
    const progressBar = document.getElementById('animProgress');

    // 设置进度条范围
    if (fromStart) {
        document.getElementById('animTimeLabel').textContent = animRange.start.toFixed(1) + 's';
        document.getElementById('animEndLabel').textContent = animRange.end.toFixed(1) + 's';
    }

    const updateProgress = (ts) => {
        const pct = ((ts - animRange.start) / duration) * 100;
        if (progressBar) progressBar.value = Math.min(100, Math.max(0, pct));
        document.getElementById('animTimeLabel').textContent = ts.toFixed(1) + 's';
        // 更新轨迹信息
        updateStats(ts);
    };

    const step = now => {
        if (!animActive) { return; }
        animElapsed = startElapsed + (now - startWall) / 1000 * animSpeed;
        const ts = Math.min(animRange.start + animElapsed, animRange.end);
        updateProgress(ts);

        if (ts >= animRange.end) {
            for (const id of playIds) {
                const pred = predictedData[id];
                if (pred?.prediction?.length) {
                    const last = pred.prediction[pred.prediction.length-1];
                    movingSpheres[id]?.position.set(last[0], last[1], last[2]);
                }
            }
            pauseAnim();
            updateProgress(animRange.end);
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
    document.getElementById('playBtn').innerHTML = '▶ 播放';
}

function stopAnim() {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    animElapsed = 0;
    const bar = document.getElementById('animProgress');
    if (bar) bar.value = 0;
    document.getElementById('playBtn').innerHTML = '▶ 播放';
    for (const id in movingSpheres) {
        scene.remove(movingSpheres[id]);
        delete movingSpheres[id];
    }
}

// ═══════════════════════════ 平台状态更新 ═══════════════════════════

function updateStats(ts) {
    for (const [id, data] of Object.entries(detectionMethods)) {
        const container = document.getElementById('stat-' + id);
        if (!container) continue;
        container.style.display = data.visible ? '' : 'none';
        if (!data.visible) continue;

        const histPts = data.points || [];
        const histTs = data.timestamps || [];
        const pred = predictedData[id];
        const predPts = pred?.prediction || [];
        const predTs = pred?.pred_times || [];

        // 合并历史+预测数据用于插值
        let allPts = [...histPts], allTs = [...histTs];
        if (predPts.length && predTs.length) {
            // 跳过预测第一个点（=历史最后点，避免重复）
            allPts = [...histPts, ...predPts.slice(1)];
            allTs = [...histTs, ...predTs.slice(1)];
        }
        if (allPts.length < 2) continue;

        const pos = lerp(allPts, allTs, Math.min(ts, allTs[allTs.length - 1]));
        const prevT = Math.max(allTs[0], ts - 0.1);
        const prevPos = lerp(allPts, allTs, prevT);
        if (!pos || !prevPos) continue;

        const dt = 0.1;
        const spd = Math.sqrt((pos[0] - prevPos[0]) ** 2 + (pos[2] - prevPos[2]) ** 2) / dt;
        const ht = pos[1];
        const angle = Math.atan2(pos[2] - prevPos[2], pos[0] - prevPos[0]) * 180 / Math.PI;

        const bolds = container.querySelectorAll('b');
        if (bolds.length >= 3) {
            bolds[0].textContent = spd.toFixed(1) + ' m/s';
            bolds[1].textContent = ht.toFixed(1) + ' m';
            bolds[2].textContent = angle.toFixed(0) + '°';
        }
    }
}

// ═══════════════════════════ 进度条拖拽 ═══════════════════════════
const animProgress = document.getElementById('animProgress');
if (animProgress) {
    animProgress.addEventListener('input', () => {
        if (animActive) pauseAnim();
        calcRange();
        const pct = parseFloat(animProgress.value) / 100;
        const duration = animRange.end - animRange.start;
        animElapsed = pct * duration;
        const ts = animRange.start + animElapsed;
        document.getElementById('animTimeLabel').textContent = ts.toFixed(1) + 's';
        updateStats(ts);

        // 首次拖拽时创建球体
        if (Object.keys(movingSpheres).length === 0) {
            const ids = new Set();
            for (const id in predictedData) ids.add(id);
            for (const [id, d] of Object.entries(detectionMethods)) {
                if (d.visible && d.points?.length >= 2) ids.add(id);
            }
            for (const id of ids) {
                const color = detectionMethods[id]?.color || '#fff';
                const sphere = makeGlowSphere(color);
                scene.add(sphere);
                movingSpheres[id] = sphere;
            }
        }

        for (const id in movingSpheres) {
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
    });
}

// ═══════════════════════════ 初始化 ═══════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initViewer();

    document.getElementById('numPointsSlider').addEventListener('input', e => {
        document.getElementById('numPointsLabel').textContent = e.target.value;
    });

    document.getElementById('predictBtn').addEventListener('click', runPrediction);

    // 播放/停止
    document.getElementById('playBtn').addEventListener('click', togglePlay);

    // 倍速
    document.getElementById('speedSelect').addEventListener('change', e => {
        animSpeed = parseFloat(e.target.value);
    });
});
