/**
 * Predict Page — 轨迹预测 + 3D 预览 + 动画播放
 */
import { toast } from '../common/toast.js';
import { lerp } from '../common/utils.js';
import { buildAxes, getSceneBackground, getFogColor, getGridColor, buildLights } from '../common/three-utils.js';
import { bindWASD, bindCoordTooltip, bindResize } from '../common/three-input.js';

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
    scene.background = new THREE.Color(getSceneBackground());
    scene.fog = new THREE.Fog(getFogColor(), 30, 120);

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
    controls.minDistance = 1;
    controls.maxDistance = 200;
    controls.autoRotate = false;

    // 地面（与总览一致）
    const gGeo = new THREE.PlaneGeometry(200, 200);
    const gMat = new THREE.MeshStandardMaterial({ color: getSceneBackground(), roughness: 0.95, metalness: 0.2, transparent: true, opacity: 0.6 });
    const plane = new THREE.Mesh(gGeo, gMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.55;
    plane.receiveShadow = true;
    scene.add(plane);

    // 网格
    const gc = getGridColor();
    const grid = new THREE.GridHelper(100, 20, gc.main, gc.sub);
    grid.position.y = -0.5;
    scene.add(grid);

    buildAxes(scene, CSS2DObject);
    buildLights(scene);
    buildRiskZone(window._PAGE_DATA_ && window._PAGE_DATA_.zone);

    function loop() {
        requestAnimationFrame(loop);
        controls.update();
        renderer.render(scene, camera);
        labelRenderer.render(scene, camera);
    }
    loop();

    bindResize(camera, renderer, labelRenderer);
    bindWASD(camera, controls, () => (window._PAGE_DATA_ && window._PAGE_DATA_.cameraSpeed) || 0.12);
    bindCoordTooltip(renderer, camera, lines, predLines, 'coordTooltip');
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            togglePlay();
        }
    });

    drawTrails();
    renderCachedPredictions();
    fitCameraToData();
    calcRange();
}

/** 首屏按当前轨迹范围自动取景（只执行一次，避免打断用户视角） */
function fitCameraToData() {
    const pts = [];
    for (const d of Object.values(detectionMethods)) {
        if (!d.visible || !d.points) continue;
        d.points.forEach(p => { if (p && p.length === 3) pts.push(p); });
    }
    if (!pts.length) return;
    const box = new THREE.Box3();
    pts.forEach(p => box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2])));
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    const r = Math.max(size.length() * 0.6, 5);
    controls.target.copy(center);
    camera.position.set(center.x + r * 1.0, center.y + r * 0.55, center.z + r * 1.25);
    camera.lookAt(center);
    controls.update();
}

// 预测段绘制: 延续该轨迹自身的视觉语言 + 终点标签（缓存恢复与实时预测共用）
function drawPrediction(mid, pred) {
    const isRecon = mid === 'synthetic';
    const color = isRecon ? RECON_COLOR : (detectionMethods[mid]?.color || '#ffffff');
    const pts = pred.prediction.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    let line;
    if (isRecon) {
        line = buildReconTrail(pred.prediction, 0.055, 0.6);   // 比历史管更细、略透明
    } else {
        line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 })
        );
    }
    if (line) scene.add(line);

    const grp = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({
        color, emissive: color, emissiveIntensity: isRecon ? 0.3 : 0.2,
        transparent: true, opacity: isRecon ? 0.65 : 0.75, roughness: 0.3,
    });
    const size = isRecon ? 0.045 : 0.05;
    pred.prediction.forEach((p, i) => {
        const s = new THREE.Mesh(new THREE.SphereGeometry(size, 10, 10), mat);
        s.position.set(p[0], p[1], p[2]);
        s.userData = { pts: pred.prediction, ts: pred.pred_times, idx: i };
        grp.add(s);
    });
    scene.add(grp);

    // 预测终点标签（与"起点"标签风格一致，便于确认预测段位置）
    const div = document.createElement('div');
    const last = pred.pred_times?.[pred.pred_times.length - 1];
    div.textContent = '预测终点' + (last != null ? ' ' + Number(last).toFixed(1) + 's' : '');
    div.style.cssText = 'color:#fff;font-size:10px;font-weight:600;background:rgba(0,0,0,0.7);'
        + 'padding:2px 8px;border-radius:10px;white-space:nowrap;';
    const label = new CSS2DObject(div);
    const end = pts[pts.length - 1];
    label.position.set(end.x, end.y, end.z);
    scene.add(label);

    predLines[mid] = { line, points: grp, label };
    return predLines[mid];
}

function renderCachedPredictions() {
    if (!Object.keys(predictedData).length) return;
    for (const [mid, pred] of Object.entries(predictedData)) {
        const m = detectionMethods[mid];
        if (!m || !m.visible) continue;
        if (!pred.prediction?.length) continue;
        drawPrediction(mid, pred);
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

const RECON_COLOR = 0x4fc3f7;

/** 综合轨迹（重建结果）: 青色管状体，与总览保持一致 */
function buildReconTrail(points, radius = 0.1, opacity = 1) {
    if (!points || points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    const segs = Math.max(150, points.length * 5);
    const group = new THREE.Group();
    group.add(new THREE.Mesh(
        new THREE.TubeGeometry(curve, segs, radius, 12, false),
        new THREE.MeshStandardMaterial({
            color: RECON_COLOR, emissive: RECON_COLOR, emissiveIntensity: 0.32,
            roughness: 0.45, metalness: 0.05, transparent: opacity < 1, opacity,
        }),
    ));
    group.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(curve.getPoints(segs)),
        new THREE.LineBasicMaterial({
            color: RECON_COLOR, transparent: true, opacity: 0.25 * opacity,
            blending: THREE.AdditiveBlending, depthWrite: false,
        }),
    ));
    return group;
}

/** 防护区（圆柱）: 与总览一致的红色细环 + 半透明侧壁 */
function buildRiskZone(zone) {
    if (!zone || !Array.isArray(zone.center) || zone.center.length !== 3) return;
    const [cx, cy, cz] = zone.center;
    const radius = Number(zone.radius) || 4;
    const y0 = Number(zone.y_min ?? 0);
    const y1 = Number(zone.y_max ?? (y0 + 20));
    const height = Math.max(y1 - y0, 1);
    const color = 0xff6b6b;
    const group = new THREE.Group();

    const wall = new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius, height, 48, 1, true),
        new THREE.MeshBasicMaterial({
            color, transparent: true, opacity: 0.02, side: THREE.DoubleSide, depthWrite: false,
        }),
    );
    wall.position.set(cx, y0 + height / 2, cz);
    group.add(wall);

    for (const y of [y0, y1]) {
        const ring = new THREE.Mesh(
            new THREE.RingGeometry(Math.max(radius - 0.03, 0.05), radius, 64),
            new THREE.MeshBasicMaterial({
                color, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false,
            }),
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(cx, y, cz);
        group.add(ring);
    }

    for (let i = 0; i < 8; i++) {
        const a = (i * Math.PI) / 4;
        const g = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(cx + radius * Math.cos(a), y0, cz + radius * Math.sin(a)),
            new THREE.Vector3(cx + radius * Math.cos(a), y1, cz + radius * Math.sin(a)),
        ]);
        group.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.22 })));
    }

    scene.add(group);
    const div = document.createElement('div');
    div.textContent = '防护区';
    div.style.cssText = 'color:#ffb3b3;font-size:10px;font-weight:600;background:rgba(0,0,0,0.65);'
        + 'padding:2px 8px;border-radius:10px;white-space:nowrap;';
    const label = new CSS2DObject(div);
    label.position.set(cx, y1, cz);
    scene.add(label);
}

function drawTrails() {
    for (const id in lines) { removeObj(lines[id]); delete lines[id]; }
    for (const id in predLines) { removeObj(predLines[id]); delete predLines[id]; }
    for (const id in startMarkers) { scene.remove(startMarkers[id]); delete startMarkers[id]; }

    for (const [id, data] of Object.entries(detectionMethods)) {
        if (!data.visible || !data.points || data.points.length < 2) continue;
        const isRecon = id === 'synthetic';
        const color = isRecon ? RECON_COLOR : data.color;

        let line;
        if (isRecon) {
            line = buildReconTrail(data.points);
        } else {
            const curve = new THREE.CatmullRomCurve3(data.points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
            line = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(curve.getPoints(100)),
                new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 }),
            );
        }
        if (line) scene.add(line);

        // 观测点：点云较密时按间隔抽样、缩小半径，突出综合轨迹
        const every = isRecon ? (data.points.length > 120 ? 2 : 1) : 4;
        const size = isRecon ? 0.07 : 0.03;
        const grp = new THREE.Group();
        const mat = new THREE.MeshStandardMaterial({
            color, emissive: color, emissiveIntensity: isRecon ? 0.3 : 0.1,
            roughness: 0.35, transparent: !isRecon, opacity: isRecon ? 1 : 0.4,
        });
        data.points.forEach((p, i) => {
            if (i % every) return;
            const s = new THREE.Mesh(new THREE.SphereGeometry(size, 10, 10), mat);
            s.position.set(p[0], p[1], p[2]);
            s.userData = { pts: data.points, ts: data.timestamps, idx: i };
            grp.add(s);
        });
        scene.add(grp);
        lines[id] = { line, points: grp };

        // 仅综合轨迹标注起终点，避免多平台标签重叠
        if (isRecon) {
            const div = document.createElement('div');
            div.textContent = '重建起点';
            div.style.cssText = 'color:#fff;font-size:10px;font-weight:600;background:rgba(0,0,0,0.7);'
                + 'padding:2px 8px;border-radius:10px;white-space:nowrap;';
            const label = new CSS2DObject(div);
            label.position.set(data.points[0][0], data.points[0][1], data.points[0][2]);
            scene.add(label);
            startMarkers[id] = label;
        }
    }
}

function removeObj(obj) {
    if (!obj) return;
    if (obj.line) scene.remove(obj.line);
    if (obj.points) scene.remove(obj.points);
    if (obj.label) scene.remove(obj.label);
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
        for (const id in predLines) { removeObj(predLines[id]); delete predLines[id]; }
        predictedData = {};

        if (result.success) {
            const results = isAll ? result.results : { [platform]: { prediction: result.prediction, pred_times: result.pred_times } };

            for (const [mid, pred] of Object.entries(results)) {
                if (!pred.prediction?.length) continue;
                predictedData[mid] = pred;

                drawPrediction(mid, pred);

                // 在平台状态下方显示预测点数
                const el = document.getElementById('pred-' + mid);
                if (el) { el.style.display = ''; el.textContent = '预测 ' + (pred.prediction.length - 1) + ' 点'; }
            }
            savePredictedData();
            calcRange();
            // 末端置信度提示（长视界为运动模型外推）
            const confEl = document.getElementById('confidenceHint');
            if (confEl) {
                const conf = result.confidence;
                if (conf && conf.length) {
                    const last = Number(conf[conf.length - 1]);
                    confEl.textContent = '末端置信度 ' + last.toFixed(2)
                        + (last < 0.5 ? ' · 长视界外推，仅供参考' : '');
                } else {
                    confEl.textContent = '';
                }
            }
            toast.success('预测完成，可播放动画查看');
        } else {
            toast.error('预测失败: ' + (result.error || ''));
        }
    } catch (e) {
        toast.error('请求失败: ' + e.message);
    } finally {
        btn.disabled = false; btn.innerHTML = '<svg width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:text-bottom;"><polygon points="9,1 3,9 7,9 6,15 13,7 9,7"/></svg> 开始预测';
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
    document.getElementById('playBtn').innerHTML = '<svg width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:text-bottom;"><rect x="3" y="2" width="3.5" height="12" rx="0.5"/><rect x="9.5" y="2" width="3.5" height="12" rx="0.5"/></svg> 暂停';

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
    document.getElementById('playBtn').innerHTML = '<svg width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:text-bottom;"><polygon points="4,2 13,8 4,14"/></svg> 播放';
    const ts = animRange.start + animElapsed;
    updateStats(ts < animRange.end ? ts : undefined);
}

function stopAnim() {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    animElapsed = 0;
    const bar = document.getElementById('animProgress');
    if (bar) bar.value = 0;
    document.getElementById('playBtn').innerHTML = '<svg width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:text-bottom;"><polygon points="4,2 13,8 4,14"/></svg> 播放';
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

/** 预测与重建联动: 默认预测重建后的综合轨迹，并显示其来源与预测视界 */
function initPipelineHint() {
    const syn = detectionMethods.synthetic;
    const select = document.getElementById('predictPlatform');
    const status = document.getElementById('reconStatus');
    if (syn && (syn.points || []).length >= 2 && select) {
        select.value = 'synthetic';
        if (status) {
            const p = syn.pipeline;
            if (p) {
                const health = (p.fusion && p.fusion.health) || {};
                const rec = p.reconstruction || {};
                status.innerHTML = '综合轨迹来自 ' + Object.keys(health).length + ' 路传感器重建 · '
                    + syn.points.length + ' 帧 · 平均不确定度 '
                    + (rec.mean_sigma != null ? Number(rec.mean_sigma).toFixed(2) + ' m' : '—');
            } else {
                status.textContent = '综合轨迹：' + syn.points.length + ' 点';
            }
        }
    }

    const slider = document.getElementById('numPointsSlider');
    const step = document.getElementById('timeStep');
    const hint = document.getElementById('horizonHint');

    // 预测参数本地记忆：刷新页面不回到默认点数/步长
    const savedPoints = localStorage.getItem('pred.numPoints');
    const savedStep = localStorage.getItem('pred.timeStep');
    if (slider && savedPoints) {
        slider.value = savedPoints;
        const label = document.getElementById('numPointsLabel');
        if (label) label.textContent = savedPoints;
    }
    if (step && savedStep) step.value = savedStep;
    const MODEL_HORIZON = 2.0;
    const update = () => {
        if (!hint) return;
        const n = parseInt(slider.value, 10) || 0;
        const dt = parseFloat(step.value) || 0;
        const horizon = n * dt;
        hint.innerHTML = '预测视界 = ' + n + ' 点 × ' + dt.toFixed(2) + ' s = ' + horizon.toFixed(1) + ' s'
            + (horizon > MODEL_HORIZON
                ? ' <span style="color:var(--text-tertiary);">（前 ' + MODEL_HORIZON.toFixed(0)
                  + ' s 为模型预测，其后为运动模型外推）</span>'
                : '');
    };
    if (slider) slider.addEventListener('input', () => {
        localStorage.setItem('pred.numPoints', slider.value);
    });
    if (step) step.addEventListener('input', () => {
        localStorage.setItem('pred.timeStep', step.value);
    });
    if (slider) slider.addEventListener('input', update);
    if (step) step.addEventListener('input', update);
    update();
}

document.addEventListener('DOMContentLoaded', () => {
    initViewer();
    initPipelineHint();

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
