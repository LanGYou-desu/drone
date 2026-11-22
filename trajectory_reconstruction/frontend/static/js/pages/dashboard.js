/**
 * Dashboard — 全屏 3D 轨迹总览
 * 高内聚：3D 场景/轨迹/动画全部封装于此
 * 低耦合：仅通过 window._PAGE_DATA_ 获取初始数据，后续通过 API 刷新
 */
import { toast } from '../common/toast.js';
import { lerp } from '../common/utils.js';
import { buildAxes, getSceneBackground, getFogColor, getGridColor, buildLights, buildStarfield } from '../common/three-utils.js';
import { bindWASD, bindCoordTooltip, bindResize } from '../common/three-input.js';

// ═══════════════════════════════════════════
// 数据加载（从 HTML 模板注入的 window._PAGE_DATA_）
// ═══════════════════════════════════════════
const { methodsData, predSettings } = window._PAGE_DATA_ || {};
if (!methodsData) {
    document.body.innerHTML = '<div style="color:#FF453A;padding:100px;text-align:center;font-size:1.5rem;">错误：未加载到数据，请检查后端服务</div>';
    throw new Error('window._PAGE_DATA_.methodsData is required');
}

let detectionMethods = JSON.parse(JSON.stringify(methodsData));
const predCfg = predSettings || { minPoints: 1, maxPoints: 20, defaultPoints: 6, defaultTimeStep: 0.5 };
let predictedTrajectories = {};
for (const id in detectionMethods) predictedTrajectories[id] = { points: [], times: [] };

// ═══════════════════════════════════════════
// Three.js 引用
// ═══════════════════════════════════════════
let scene, camera, renderer, labelRenderer, controls, THREE, CSS2DObject;
let lines = {}, predLines = {}, movingSpheres = {}, predictedSpheres = {};
let trailSystems = {};
let animActive = false, animId = null, animSpeed = 1.0;
let animElapsed = 0;   // 已播放的时间（秒），暂停时保留
let zoneGroup = null;  // 防护区（3D）
let timeRange = { start: 0, end: 0 };

// ═══════════════════════════════════════════
// 场景初始化
// ═══════════════════════════════════════════
async function init() {
    try {
        THREE = await import('three');
        const orb = await import('three/addons/controls/OrbitControls.js');
        const css = await import('three/addons/renderers/CSS2DRenderer.js');
        CSS2DObject = css.CSS2DObject;

        buildScene(THREE, orb, css);
        buildLights(scene);
        buildGround();
        buildGrid();
        buildStarfield(scene);
        buildRiskZone(window._PAGE_DATA_ && window._PAGE_DATA_.zone);

        startLoop(orb, css);
        refreshAll();
        fitCameraToData();
        calcRange();
        bindEvents();
        bindWASD(camera, controls, () => (window._PAGE_DATA_ && window._PAGE_DATA_.cameraSpeed) || 0.12);
        bindKeyboard();

        toast.success('就绪 — 左键旋转 | 滚轮缩放 | 右键平移 | 空格播放');
    } catch (e) {
        console.error('Dashboard init failed:', e);
        toast.error('3D 初始化失败: ' + e.message);
    }
}

/** 首屏按当前轨迹范围自动取景（仅初始化时执行一次，避免打断用户视角） */
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

function buildScene(THREE, orb, css) {
    scene = new THREE.Scene();
    const bg = getSceneBackground();
    scene.background = new THREE.Color(bg);
    scene.fog = new THREE.Fog(getFogColor(), 30, 120);

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
    camera.position.set(12, 7, 14);
    camera.lookAt(4, 2, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    document.getElementById('viewer').appendChild(renderer.domElement);

    labelRenderer = new css.CSS2DRenderer();
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.domElement.style.cssText = 'position:absolute;top:0;pointer-events:none;';
    document.getElementById('viewer').appendChild(labelRenderer.domElement);

    controls = new orb.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(4, 2, 4);
    controls.minDistance = 1;
    controls.maxDistance = 200;
    controls.autoRotate = false;
}


function buildGround() {
    const geo = new THREE.PlaneGeometry(200, 200);
    const mat = new THREE.MeshStandardMaterial({ color: getSceneBackground(), roughness: 0.95, metalness: 0.2, transparent: true, opacity: 0.6 });
    const plane = new THREE.Mesh(geo, mat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.55;
    plane.receiveShadow = true;
    scene.add(plane);
}

/** 防护区（圆柱）: 红色细环 + 半透明侧壁 + 母线，风格与轨迹/网格一致 */
function buildRiskZone(zone) {
    if (!zone || !Array.isArray(zone.center) || zone.center.length !== 3) return;
    const [cx, cy, cz] = zone.center;
    const radius = Number(zone.radius) || 4;
    const y0 = Number(zone.y_min ?? 0);
    const y1 = Number(zone.y_max ?? (y0 + 20));
    const height = Math.max(y1 - y0, 1);
    const color = 0xff6b6b;
    const group = new THREE.Group();

    group.add(new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius, height, 48, 1, true),
        new THREE.MeshBasicMaterial({
            color, transparent: true, opacity: 0.02, side: THREE.DoubleSide, depthWrite: false,
        }),
    ).translateY(0));

    const wall = group.children[group.children.length - 1];
    wall.position.set(cx, y0 + height / 2, cz);

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
    const label = addLabel('防护区', [cx, y1, cz], 'color:#ffb3b3;font-size:10px;font-weight:600;'
        + 'background:rgba(0,0,0,0.65);padding:2px 8px;border-radius:10px;white-space:nowrap;');
    scene.add(label);
    zoneGroup = group;
}

function buildGrid() {
    // 主网格 — 清晰 5m 间距
    const gc = getGridColor();
    const grid = new THREE.GridHelper(100, 20, gc.main, gc.sub);
    grid.position.y = -0.5;
    scene.add(grid);

    buildAxes(scene, CSS2DObject);
}


function startLoop(orb, css) {
    function loop() {
        requestAnimationFrame(loop);
        controls.update();
        updateAllTrails();
        renderer.render(scene, camera);
        labelRenderer.render(scene, camera);
    }
    loop();
}

// ═══════════════════════════════════════════
// 轨迹渲染
// ═══════════════════════════════════════════

const RECON_COLOR = 0x4fc3f7;

/** 综合轨迹（重建结果）: 青色管状体 + 柔和外发光，与"重建"视觉语言一致 */
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

function buildTrailMesh(points, color) {
    if (!points || points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    const curvePts = curve.getPoints(Math.max(100, points.length * 5));

    const group = new THREE.Group();

    // 外发光
    const gGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
    group.add(new THREE.Line(gGeo, new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.15, blending: THREE.AdditiveBlending, depthWrite: false,
    })));

    // 主线（作为背景证据，弱化以突出综合重建轨迹）
    const lGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
    group.add(new THREE.Line(lGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 })));

    return group;
}

function buildSpheres(points, color, size = 0.06, opacity = 1, timestamps = null, every = 1) {
    const group = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.15, roughness: 0.3, transparent: opacity < 1, opacity });
    points.forEach((p, i) => {
        if (every > 1 && i % every !== 0) return;
        const s = new THREE.Mesh(new THREE.SphereGeometry(size, 10, 10), mat);
        s.position.set(p[0], p[1], p[2]);
        s.castShadow = true;
        s.userData = { pts: points, ts: timestamps, idx: i };
        group.add(s);
    });
    return group;
}

function addLabel(text, pos, style) {
    const div = document.createElement('div');
    div.textContent = text;
    div.style.cssText = style;
    const label = new CSS2DObject(div);
    label.position.set(pos[0], pos[1] + 0.35, pos[2]);
    return label;
}

const LABEL_CSS = 'color:#fff;font-size:10px;font-weight:600;background:rgba(0,0,0,0.7);padding:2px 8px;border-radius:10px;backdrop-filter:blur(8px);';

function refreshAll() {
    // 清理旧元素
    for (const id in lines) removeTrail(id);
    for (const id in predLines) removePred(id);
    for (const id in movingSpheres) { scene.remove(movingSpheres[id]); delete movingSpheres[id]; }
    for (const id in predictedSpheres) { scene.remove(predictedSpheres[id]); delete predictedSpheres[id]; }

    for (const [id, data] of Object.entries(detectionMethods)) {
        if (!data.visible || !data.points || data.points.length < 2) continue;

        const isRecon = id === 'synthetic';
        const density = data.points.length > 120 ? 2 : 1;
        const trail = isRecon ? buildReconTrail(data.points) : buildTrailMesh(data.points, data.color);
        const spheres = buildSpheres(data.points, isRecon ? RECON_COLOR : data.color,
                                     isRecon ? 0.07 : 0.03,
                                     isRecon ? 1 : 0.40, data.timestamps,
                                     isRecon ? Math.max(density, 2) : 4);
        const pts = data.points;

        lines[id] = { trail, spheres };
        if (trail) scene.add(trail);
        scene.add(spheres);
        if (isRecon) {   // 只给综合轨迹加起终点标签，避免多平台标签重叠
            const startLab = addLabel('重建起点', pts[0], LABEL_CSS);
            const endLab = addLabel('重建终点', pts[pts.length-1], LABEL_CSS);
            lines[id].startLab = startLab;
            lines[id].endLab = endLab;
            scene.add(startLab); scene.add(endLab);
        }

        // 预测线
        const pred = predictedTrajectories[id];
        if (pred?.points?.length >= 2) {
            const lastPt = data.points?.length ? data.points[data.points.length - 1] : null;
            addPredLine(id, pred, isRecon ? '#4fc3f7' : data.color, lastPt);
        }
    }
    renderReconSummary();
}

/** 左侧"综合重建"面板: 展示管线质量指标 */
function renderReconSummary() {
    const el = document.getElementById('reconSummary');
    if (!el) return;
    const syn = detectionMethods.synthetic;
    const pipeline = syn && syn.pipeline;
    if (!pipeline) {
        el.innerHTML = '<span style="color:var(--text-tertiary);">尚无重建结果</span>';
        return;
    }
    const rec = pipeline.reconstruction || {};
    const fusion = pipeline.fusion || {};
    const health = fusion.health || {};
    const total = Object.keys(health).length;
    const selected = (fusion.selected || []).length;
    const excluded = fusion.excluded || [];
    const row = (k, v) => '<div style="display:flex;justify-content:space-between;gap:8px;">'
        + '<span>' + k + '</span>'
        + '<b style="color:var(--text-primary);font-family:var(--mono);">' + v + '</b></div>';
    el.innerHTML =
        row('轨迹帧数', (syn.points || []).length + ' 帧') +
        row('参与传感器', selected + ' / ' + total) +
        (excluded.length ? row('已排除', excluded.join('、')) : '') +
        row('融合不确定度', rec.mean_sigma != null ? Number(rec.mean_sigma).toFixed(2) + ' m' : '—') +
        row('重建置信度', rec.mean_confidence != null ? Number(rec.mean_confidence).toFixed(2) : '—') +
        row('补洞 / 剔除', (rec.n_filled || 0) + ' / ' + (rec.n_outliers || 0) + ' 帧');
}

async function rebuildSynthesis() {
    const btn = document.getElementById('rebuildBtn');
    if (!btn) return;
    btn.disabled = true;
    const old = btn.textContent;
    btn.textContent = '重建中…';
    try {
        const payload = {
            hz: Number(document.getElementById('reconHz')?.value) || undefined,
            gate_m: Number(document.getElementById('reconGate')?.value) || undefined,
            smooth_window: Number(document.getElementById('reconSmooth')?.value) || undefined,
            max_gap_s: Number(document.getElementById('reconGap')?.value) || undefined,
        };
        const resp = await fetch('/api/synthesize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const r = await resp.json();
        if (r.success) {
            location.reload();
        } else {
            toast.error(r.error || '重建失败');
        }
    } catch (e) {
        toast.error('重建请求失败');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

function addPredLine(id, pred, color, lastHistPt) {
    // 预测段沿用该轨迹自身的视觉语言：
    //   综合轨迹 → 更细、略透明的青色管（延续重建轨迹）；
    //   单平台   → 同色实线 + 与历史点同尺寸的小球。
    const isRecon = id === 'synthetic';
    const allPts = pred.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    let line;
    if (isRecon) {
        line = buildReconTrail(pred.points, 0.055, 0.6);
    } else {
        line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(allPts),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 })
        );
    }

    const grp = buildSpheres(pred.points, isRecon ? RECON_COLOR : color,
                             isRecon ? 0.045 : 0.05, isRecon ? 0.6 : 0.75, pred.times, 1);
    if (line) scene.add(line);
    scene.add(grp);
    predLines[id] = { line, points: grp };
}

function removeTrail(id) {
    const o = lines[id]; if (!o) return;
    if (o.trail) scene.remove(o.trail);
    scene.remove(o.spheres);
    scene.remove(o.startLab); scene.remove(o.endLab);
    delete lines[id];
}

function removePred(id) {
    const o = predLines[id]; if (!o) return;
    scene.remove(o.line); scene.remove(o.points);
    delete predLines[id];
}

// ═══════════════════════════════════════════
// 拖尾粒子
// ═══════════════════════════════════════════

function ensureTrail(id, color) {
    if (trailSystems[id]) return;
    const count = 2000;
    const geo = new THREE.BufferGeometry();
    const pa = new Float32Array(count * 3).fill(-1000);
    const ca = new Float32Array(count * 3).fill(0);
    geo.setAttribute('position', new THREE.BufferAttribute(pa, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(ca, 3));
    const sys = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.05, vertexColors: true, transparent: true, opacity: 0.7,
        blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(sys);
    trailSystems[id] = { sys, particles: Array.from({length:count}, () => ({pos:new THREE.Vector3(), life:0, color:new THREE.Color(color)})), next: 0 };
}

function emitParticle(id, pos, color) {
    ensureTrail(id, color);
    const tp = trailSystems[id];
    const p = tp.particles[tp.next];
    p.pos.copy(pos); p.life = 1; p.color.set(color);
    tp.next = (tp.next + 1) % tp.particles.length;
}

function updateAllTrails() {
    for (const id in trailSystems) {
        const tp = trailSystems[id];
        const pa = tp.sys.geometry.attributes.position.array;
        const ca = tp.sys.geometry.attributes.color.array;
        for (let i = 0; i < tp.particles.length; i++) {
            const p = tp.particles[i];
            if (p.life > 0) {
                p.life -= 0.01;
                pa[i*3]=p.pos.x; pa[i*3+1]=p.pos.y; pa[i*3+2]=p.pos.z;
                ca[i*3]=p.color.r*p.life; ca[i*3+1]=p.color.g*p.life; ca[i*3+2]=p.color.b*p.life;
            } else { pa[i*3]=pa[i*3+1]=pa[i*3+2]=-1000; }
        }
        tp.sys.geometry.attributes.position.needsUpdate = true;
        tp.sys.geometry.attributes.color.needsUpdate = true;
    }
}

// ═══════════════════════════════════════════
// 时间轴动画
// ═══════════════════════════════════════════

function calcRange() {
    let min = Infinity, max = -Infinity;
    for (const id in detectionMethods) {
        const ts = detectionMethods[id]?.timestamps;
        if (ts?.length) { min = Math.min(min, ts[0]); max = Math.max(max, ts[ts.length-1]); }
        const pred = predictedTrajectories[id];
        if (pred?.times?.length) max = Math.max(max, pred.times[pred.times.length-1]);
    }
    timeRange = { start: min===Infinity?0:min, end: max===-Infinity?0:max };
    const s = document.getElementById('timeStart'), e = document.getElementById('timeEnd');
    if (s) s.textContent = timeRange.start.toFixed(1) + 's';
    if (e) e.textContent = timeRange.end.toFixed(1) + 's';
}

function setPredVisible(v) {
    for (const id in predLines) {
        if (predLines[id]?.line) predLines[id].line.visible = v;
        if (predLines[id]?.points) predLines[id].points.visible = v;
    }
}

function animStep(ts) {
    for (const id in detectionMethods) {
        const data = detectionMethods[id], pred = predictedTrajectories[id];
        if (!data.visible) {
            // 平台被隐藏时，播放球也必须隐藏（否则隐藏轨迹上仍有目标点）
            if (movingSpheres[id]) movingSpheres[id].visible = false;
            if (predictedSpheres[id]) predictedSpheres[id].visible = false;
            continue;
        }
        if (data.points?.length && data.timestamps?.length && ts <= data.timestamps[data.timestamps.length-1]) {
            const pos = lerp(data.points, data.timestamps, ts);
            if (pos) { ensureMovingSphere(id, data.color, 0.18); movingSpheres[id].position.set(pos[0],pos[1],pos[2]); movingSpheres[id].visible = true; emitParticle(id, movingSpheres[id].position, data.color); }
            if (predictedSpheres[id]) predictedSpheres[id].visible = false;
        } else if (pred?.points?.length && pred?.times?.length && ts <= pred.times[pred.times.length-1]) {
            const pos = lerp(pred.points, pred.times, ts);
            if (pos) { ensurePredSphere(id, data.color, 0.12); predictedSpheres[id].position.set(pos[0],pos[1],pos[2]); predictedSpheres[id].visible = true; emitParticle(id, predictedSpheres[id].position, data.color); }
            if (movingSpheres[id]) movingSpheres[id].visible = false;
        }
    }
    updateStats(ts);
}

function ensureMovingSphere(id, color, size) {
    if (movingSpheres[id]) return;
    const g = new THREE.Group();
    g.add(new THREE.Mesh(new THREE.SphereGeometry(size,16,16), new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:0.8,roughness:0.2})));
    g.add(new THREE.Mesh(new THREE.SphereGeometry(size*2,12,12), new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.2,blending:THREE.AdditiveBlending,depthWrite:false})));
    scene.add(g); movingSpheres[id] = g;
}

function ensurePredSphere(id, color, size) {
    if (predictedSpheres[id]) return;
    const g = new THREE.Group();
    g.add(new THREE.Mesh(new THREE.SphereGeometry(size,12,12), new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:0.5,transparent:true,opacity:0.7})));
    scene.add(g); predictedSpheres[id] = g;
}

function togglePlay() {
    if (animActive) { pauseAnim(); return; }
    startAnim(animElapsed >= timeRange.end - timeRange.start);
}

function startAnim(fromStart = false) {
    if (fromStart) animElapsed = 0;  // 重播时从头开始
    if (animActive) return;
    calcRange();
    if (timeRange.end <= timeRange.start) { toast.warning('无有效时间数据'); return; }
    if (animElapsed >= timeRange.end - timeRange.start) animElapsed = 0;  // 已播完则重置

    animActive = true;
    const btn = document.getElementById('playBtn');
    if (btn) btn.innerHTML = '<svg width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:text-bottom;"><rect x="3" y="2" width="3.5" height="12" rx="0.5"/><rect x="9.5" y="2" width="3.5" height="12" rx="0.5"/></svg> 暂停';
    setPredVisible(false);
    const startWall = performance.now();
    const startElapsed = animElapsed;  // 从暂停点开始

    const step = now => {
        if (!animActive) {
            animElapsed = startElapsed + (now - startWall) / 1000 * animSpeed;
            return;  // 被暂停时保存 elapsed
        }
        animElapsed = startElapsed + (now - startWall) / 1000 * animSpeed;
        const ts = timeRange.start + animElapsed;

        if (ts >= timeRange.end) {
            animStep(timeRange.end);
            pauseAnim();
            return;
        }

        animStep(ts);
        const slider = document.getElementById('timelineSlider');
        if (slider) slider.value = ((ts - timeRange.start) / (timeRange.end - timeRange.start)) * 100;
        const cur = document.getElementById('timeCurrent');
        if (cur) cur.textContent = ts.toFixed(1) + 's';
        animId = requestAnimationFrame(step);
    };
    animId = requestAnimationFrame(step);
}

function pauseAnim() {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    const btn = document.getElementById('playBtn');
    if (btn) btn.innerHTML = '<svg width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:text-bottom;"><polygon points="4,2 13,8 4,14"/></svg> 播放';
    setPredVisible(false);
    const ts = timeRange.start + animElapsed;
    updateStats(ts < timeRange.end ? ts : undefined);
}

function stopAnim() {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    animElapsed = 0;
    setPredVisible(true);
    // 清除动画球体
    for (const id in movingSpheres) { scene.remove(movingSpheres[id]); delete movingSpheres[id]; }
    for (const id in predictedSpheres) { scene.remove(predictedSpheres[id]); delete predictedSpheres[id]; }
    refreshAll();
    updateStats();
    const slider = document.getElementById('timelineSlider');
    if (slider) slider.value = 0;
}

// ═══════════════════════════════════════════
// UI 更新
// ═══════════════════════════════════════════

function updateStats(ts) {
    for (const [id, data] of Object.entries(detectionMethods)) {
        const container = document.getElementById('stat-' + id);
        if (!container) continue;
        container.style.display = data.visible ? '' : 'none';
        if (!data.visible) continue;

        const pts = data.points || [];
        const tsArr = data.timestamps || [];
        if (pts.length < 2) continue;

        let pos, prevPos;
        if (ts !== undefined && tsArr.length) {
            pos = lerp(pts, tsArr, Math.min(ts, tsArr[tsArr.length - 1]));
            prevPos = lerp(pts, tsArr, Math.max(tsArr[0], ts - 0.1));
        }
        pos = pos || pts[pts.length - 1];
        prevPos = prevPos || (pts.length >= 2 ? pts[pts.length - 2] : pos);

        const spd = Math.sqrt((pos[0] - prevPos[0]) ** 2 + (pos[2] - prevPos[2]) ** 2) / 0.1;
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

// 暴露给 HTML onclick
window.toggleMethod = function(id) {
    detectionMethods[id].visible = !detectionMethods[id].visible;
    const badge = document.getElementById('status-' + id);
    if (badge) { badge.className = 'badge ' + (detectionMethods[id].visible ? 'badge-green' : 'badge-red'); badge.textContent = detectionMethods[id].visible ? '显示' : '隐藏'; }
    // 同步显示/隐藏平台状态面板
    const statPanel = document.getElementById('stat-' + id);
    if (statPanel) statPanel.style.display = detectionMethods[id].visible ? '' : 'none';
    refreshAll(); updateStats();
    // 同步到服务端
    fetch('/api/toggle_method', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method_id: id }),
    }).catch(function(e){ console.warn('toggleMethod sync failed:', e); });
};

// ═══════════════════════════════════════════
// 事件绑定
// ═══════════════════════════════════════════

function bindEvents() {
    document.getElementById('playBtn')?.addEventListener('click', togglePlay);
    document.getElementById('speedSelect')?.addEventListener('change', e => { animSpeed = parseFloat(e.target.value); });
    document.getElementById('rebuildBtn')?.addEventListener('click', rebuildSynthesis);
    document.getElementById('timelineSlider')?.addEventListener('input', e => {
        if (animActive) pauseAnim();
        setPredVisible(false);
        calcRange();
        if (timeRange.end > timeRange.start) {
            const ts = timeRange.start + parseFloat(e.target.value)/100*(timeRange.end-timeRange.start);
            animStep(ts);
                animElapsed = ts - timeRange.start;
            updateStats(ts);
            const cur = document.getElementById('timeCurrent');
            if (cur) cur.textContent = ts.toFixed(1) + 's';
        }
    });

    bindCoordTooltip(renderer, camera, lines, predLines, 'coordTooltip');
    bindResize(camera, renderer, labelRenderer);
}


function bindKeyboard() {
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            animActive ? pauseAnim() : startAnim(false);
        }
    });
}

// ═══════════════════════════════════════════
// 启动
// ═══════════════════════════════════════════
init();
