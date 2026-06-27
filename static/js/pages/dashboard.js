/**
 * Dashboard — 全屏 3D 轨迹总览
 * 高内聚：3D 场景/轨迹/动画全部封装于此
 * 低耦合：仅通过 window._PAGE_DATA_ 获取初始数据，后续通过 API 刷新
 */
import { toast } from '../common/toast.js';

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
let animActive = false, animId = null, animSpeed = 1.0, animStart = 0;
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
        buildLights();
        buildGround();
        buildGrid();
        buildStarfield();

        startLoop(orb, css);
        refreshAll();
        updateStats();
        bindEvents();
        bindKeyboard();

        toast.success('就绪 — 拖拽旋转 | 滚轮缩放 | 右键平移 | 空格播放');
    } catch (e) {
        console.error('Dashboard init failed:', e);
        toast.error('3D 初始化失败: ' + e.message);
    }
}

function buildScene(THREE, orb, css) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);
    scene.fog = new THREE.FogExp2(0x0a0a14, 0.0002);

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500);
    camera.position.set(14, 8, 16);
    camera.lookAt(4, 2, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    document.getElementById('viewer').appendChild(renderer.domElement);

    labelRenderer = new css.CSS2DRenderer();
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.domElement.style.cssText = 'position:absolute;top:0;pointer-events:none;';
    document.getElementById('viewer').appendChild(labelRenderer.domElement);

    controls = new orb.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(4, 2, 4);
    controls.minDistance = 2;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI * 0.65;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.2;
}

function buildLights() {
    scene.add(new THREE.AmbientLight(0x334466, 0.8));
    scene.add(new THREE.HemisphereLight(0x8899cc, 0x223344, 0.5));
    const sun = new THREE.DirectionalLight(0xffeedd, 1.2);
    sun.position.set(12, 20, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 100;
    sun.shadow.camera.left = -30; sun.shadow.camera.right = 30;
    sun.shadow.camera.top = 30; sun.shadow.camera.bottom = -30;
    sun.shadow.bias = -0.0001;
    scene.add(sun);
    scene.add(new THREE.DirectionalLight(0x4466aa, 0.3).position.set(-5, 3, -5));
}

function buildGround() {
    const geo = new THREE.PlaneGeometry(200, 200);
    const mat = new THREE.MeshStandardMaterial({ color: 0x1a1a28, roughness: 0.95, metalness: 0.2, transparent: true, opacity: 0.6 });
    const plane = new THREE.Mesh(geo, mat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.55;
    plane.receiveShadow = true;
    scene.add(plane);
}

function buildGrid() {
    // 细网格
    const fine = new THREE.GridHelper(200, 200, 0x222244, 0x111122);
    fine.position.y = -0.53;
    scene.add(fine);

    // 粗网格（10m 间距）
    const coarse = new THREE.GridHelper(200, 20, 0x4466aa, 0x111122);
    coarse.position.y = -0.52;
    coarse.material.transparent = true;
    coarse.material.opacity = 0.6;
    scene.add(coarse);

    // ★ 标准坐标轴（之前缺失导致看不见）★
    scene.add(new THREE.AxesHelper(10));
}

function buildStarfield() {
    const count = 4000;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        pos[i*3] = (Math.random() - 0.5) * 160;
        pos[i*3+1] = Math.random() * 50 + 1;
        pos[i*3+2] = (Math.random() - 0.5) * 160;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
        color: 0x6688cc, size: 0.05, transparent: true, opacity: 0.3,
        blending: THREE.AdditiveBlending, depthWrite: false,
    })));
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

function buildTrailMesh(points, color) {
    if (!points || points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    const curvePts = curve.getPoints(Math.max(100, points.length * 5));

    const group = new THREE.Group();

    // 外发光
    const gGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
    group.add(new THREE.Line(gGeo, new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending, depthWrite: false,
    })));

    // 主线
    const lGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
    group.add(new THREE.Line(lGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 })));

    return group;
}

function buildSpheres(points, color, size = 0.06, opacity = 1) {
    const group = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4, roughness: 0.25, transparent: opacity < 1, opacity });
    points.forEach(p => {
        const s = new THREE.Mesh(new THREE.SphereGeometry(size, 10, 10), mat);
        s.position.set(p[0], p[1], p[2]);
        s.castShadow = true;
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

        const trail = buildTrailMesh(data.points, data.color);
        const spheres = buildSpheres(data.points, data.color, 0.06);
        const pts = data.points;
        const startLab = addLabel('🚀 起点', pts[0], LABEL_CSS);
        const endLab = addLabel('🏁 终点', pts[pts.length-1], LABEL_CSS);
        const nameLab = addLabel(data.name, pts[Math.floor(pts.length/2)],
            `color:${data.color};font-size:12px;font-weight:700;background:rgba(0,0,0,0.75);padding:4px 12px;border-radius:14px;backdrop-filter:blur(8px);border:1px solid ${data.color}44;`);

        lines[id] = { trail, spheres, startLab, endLab, nameLab };
        if (trail) scene.add(trail);
        scene.add(spheres);
        scene.add(startLab); scene.add(endLab); scene.add(nameLab);

        // 预测线
        const pred = predictedTrajectories[id];
        if (pred?.points?.length >= 2) {
            addPredLine(id, pred, data.color);
        }
    }
}

function addPredLine(id, pred, color) {
    const pts = pred.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineDashedMaterial({ color, dashSize: 0.35, gapSize: 0.2, transparent: true, opacity: 0.55 })
    );
    line.computeLineDistances();
    const grp = buildSpheres(pred.points, color, 0.04, 0.5);
    scene.add(line); scene.add(grp);
    predLines[id] = { line, points: grp };
}

function removeTrail(id) {
    const o = lines[id]; if (!o) return;
    if (o.trail) scene.remove(o.trail);
    scene.remove(o.spheres);
    scene.remove(o.startLab); scene.remove(o.endLab); scene.remove(o.nameLab);
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

function lerp(pts, times, t) {
    if (!pts?.length || !times?.length) return null;
    if (t <= times[0]) return [...pts[0]];
    if (t >= times[times.length-1]) return [...pts[times.length-1]];
    let i = 0;
    while (i < times.length-1 && times[i+1] < t) i++;
    const r = (t - times[i]) / (times[i+1] - times[i]);
    return [pts[i][0]+(pts[i+1][0]-pts[i][0])*r, pts[i][1]+(pts[i+1][1]-pts[i][1])*r, pts[i][2]+(pts[i+1][2]-pts[i][2])*r];
}

function calcRange() {
    let min = Infinity, max = -Infinity;
    for (const id in detectionMethods) {
        const ts = detectionMethods[id]?.timestamps;
        if (ts?.length) { min = Math.min(min, ts[0]); max = Math.max(max, ts[ts.length-1]); }
        const pred = predictedTrajectories[id];
        if (pred?.times?.length) max = Math.max(max, pred.times[pred.times.length-1]);
    }
    timeRange = { start: min===Infinity?0:min, end: max===-Infinity?0:max };
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

function startAnim() {
    if (animActive) return;
    calcRange();
    if (timeRange.end <= timeRange.start) { toast.warning('无有效时间数据'); return; }
    animActive = true; animStart = performance.now();
    setPredVisible(false);
    const step = now => {
        if (!animActive) return;
        const ts = timeRange.start + (now - animStart)/1000*animSpeed;
        if (ts >= timeRange.end) { stopAnim(true); animStep(timeRange.end); return; }
        animStep(ts);
        const slider = document.getElementById('timelineSlider');
        if (slider) slider.value = ((ts-timeRange.start)/(timeRange.end-timeRange.start))*100;
        animId = requestAnimationFrame(step);
    };
    animId = requestAnimationFrame(step);
}

function stopAnim(restore=true) {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    if (restore) { setPredVisible(true); refreshAll(); }
}

// ═══════════════════════════════════════════
// UI 更新
// ═══════════════════════════════════════════

function updateStats() {
    let total = 0;
    for (const id in detectionMethods) total += (detectionMethods[id].points || []).length;
    const tpEl = document.getElementById('totalPoints');
    if (tpEl) tpEl.textContent = total;

    for (const [id, data] of Object.entries(detectionMethods)) {
        if (!data.visible || data.points?.length < 2) continue;
        const pts = data.points, ts = data.timestamps || [], l = pts.length;
        const dt = ts[l-1] - ts[l-2] || 0.5;
        const spd = Math.sqrt((pts[l-1][0]-pts[l-2][0])**2 + (pts[l-1][2]-pts[l-2][2])**2) / dt;
        const spdEl = document.getElementById('currentSpeed');
        const htEl = document.getElementById('currentHeight');
        if (spdEl) spdEl.textContent = spd.toFixed(1) + ' m/s';
        if (htEl) htEl.textContent = pts[l-1][1].toFixed(1) + ' m';
        break;
    }
    const luEl = document.getElementById('lastUpdate');
    if (luEl) luEl.textContent = new Date().toLocaleTimeString('zh-CN');
}

// 暴露给 HTML onclick
window.toggleMethod = function(id) {
    detectionMethods[id].visible = !detectionMethods[id].visible;
    const badge = document.getElementById('status-' + id);
    if (badge) { badge.className = 'badge ' + (detectionMethods[id].visible ? 'badge-green' : 'badge-red'); badge.textContent = detectionMethods[id].visible ? '显示' : '隐藏'; }
    refreshAll(); updateStats();
};

// ═══════════════════════════════════════════
// 事件绑定
// ═══════════════════════════════════════════

function bindEvents() {
    document.getElementById('playBtn')?.addEventListener('click', startAnim);
    document.getElementById('stopBtn')?.addEventListener('click', () => stopAnim(true));
    document.getElementById('speedSelect')?.addEventListener('change', e => { animSpeed = parseFloat(e.target.value); });
    document.getElementById('timelineSlider')?.addEventListener('input', e => {
        if (animActive) stopAnim(false);
        setPredVisible(false);
        calcRange();
        if (timeRange.end > timeRange.start) {
            animStep(timeRange.start + parseFloat(e.target.value)/100*(timeRange.end-timeRange.start));
        }
    });

    // 鼠标坐标
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    renderer.domElement.addEventListener('mousemove', e => {
        mouse.x = (e.clientX / renderer.domElement.clientWidth) * 2 - 1;
        mouse.y = -(e.clientY / renderer.domElement.clientHeight) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const targets = [];
        for (const id in lines) { if (lines[id]?.spheres) lines[id].spheres.children.forEach(s => targets.push(s)); }
        const hits = raycaster.intersectObjects(targets);
        const tip = document.getElementById('coordTooltip');
        if (hits.length > 0 && tip) {
            const p = hits[0].point;
            tip.innerHTML = `📍 <span style="color:#0A84FF">X</span>${p.x.toFixed(2)} <span style="color:#30D158">Y</span>${p.y.toFixed(2)} <span style="color:#FF9F0A">Z</span>${p.z.toFixed(2)}`;
            tip.style.display = 'block';
            tip.style.left = (e.clientX + 14) + 'px';
            tip.style.top = (e.clientY - 24) + 'px';
        } else if (tip) { tip.style.display = 'none'; }
    });

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
    });
}

function bindKeyboard() {
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            animActive ? stopAnim(true) : startAnim();
        }
    });
}

// ═══════════════════════════════════════════
// 启动
// ═══════════════════════════════════════════
init();
