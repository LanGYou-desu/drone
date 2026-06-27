/**
 * Dashboard — 全屏 3D 轨迹总览
 * 高质量投影：渐变背景、软阴影、发光轨迹、深度雾、粒子场
 */
import { toast } from '../common/toast.js';

// ---- 后端注入数据 ----
const methodsData = {{ methods_data | tojson }};
const predCfg = {{ pred_settings | tojson }};
let detectionMethods = JSON.parse(JSON.stringify(methodsData));
let predictedTrajectories = {};
for (const id in detectionMethods) predictedTrajectories[id] = { points: [], times: [] };

// ---- Three.js 对象 ----
let scene, camera, renderer, labelRenderer, controls, THREE, CSS2DObject;
let lines = {}, predLines = {}, movingSpheres = {}, predictedSpheres = {};
let trailSystems = {};
let animActive = false, animId = null, animSpeed = 1.0, animStart = 0;
let timeRange = { start: 0, end: 0 };

// ---- 初始化场景 ----
async function init() {
    THREE = await import('three');
    const orb = await import('three/addons/controls/OrbitControls.js');
    const css = await import('three/addons/renderers/CSS2DRenderer.js');
    CSS2DObject = css.CSS2DObject;

    // ── 场景 ──
    scene = new THREE.Scene();

    // 渐变背景（底部深色 → 顶部微亮）
    const bgCanvas = document.createElement('canvas');
    bgCanvas.width = 2; bgCanvas.height = 512;
    const ctx = bgCanvas.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, 512);
    grad.addColorStop(0, '#1a1a2e');
    grad.addColorStop(0.5, '#0f0f18');
    grad.addColorStop(1, '#080810');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, 2, 512);
    const bgTex = new THREE.CanvasTexture(bgCanvas);
    scene.background = bgTex;
    scene.fog = new THREE.FogExp2(0x080810, 0.00025);

    // ── 相机 ──
    camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 500);
    camera.position.set(16, 11, 18);
    camera.lookAt(5, 3, 5);

    // ── 渲染器（抗锯齿 + 阴影） ──
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    document.getElementById('viewer').appendChild(renderer.domElement);

    // ── 标签渲染器 ──
    labelRenderer = new css.CSS2DRenderer();
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.domElement.style.cssText = 'position:absolute;top:0;pointer-events:none;';
    document.getElementById('viewer').appendChild(labelRenderer.domElement);

    // ── 控制器（带阻尼+惯性） ──
    controls = new orb.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(5, 3, 5);
    controls.minDistance = 3;
    controls.maxDistance = 80;
    controls.maxPolarAngle = Math.PI * 0.65;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.15;

    // ── 地面 ──
    addGround();
    // ── 灯光 ──
    addLights();
    // ── 粒子场 ──
    addStarfield();
    // ── 网格 ──
    addGrid();

    // ── 渲染循环 ──
    function loop() {
        requestAnimationFrame(loop);
        controls.update();
        updateAllTrails();
        renderer.render(scene, camera);
        labelRenderer.render(scene, camera);
    }
    loop();

    // ── 事件 ──
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    window.addEventListener('resize', onResize);

    refreshAll();
    updateRealtimeStats();
    bindUI();

    toast.success('就绪 — 拖拽旋转 | 滚轮缩放 | 右键平移');
}

// ============================================================
// 场景元素
// ============================================================

function addGround() {
    // 大型半透明地平面
    const geo = new THREE.PlaneGeometry(300, 300);
    const mat = new THREE.MeshStandardMaterial({
        color: 0x1a1a2e,
        roughness: 0.9,
        metalness: 0.3,
        transparent: true,
        opacity: 0.7,
    });
    const plane = new THREE.Mesh(geo, mat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.6;
    plane.receiveShadow = true;
    scene.add(plane);
}

function addLights() {
    // 环境光（暗蓝调）
    scene.add(new THREE.AmbientLight(0x334466, 0.7));

    // 半球光（天空蓝 + 地面暗）
    scene.add(new THREE.HemisphereLight(0x8899cc, 0x223344, 0.5));

    // 主方向光（带阴影）
    const sun = new THREE.DirectionalLight(0xffeedd, 1.3);
    sun.position.set(15, 25, 10);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 120;
    sun.shadow.camera.left = -40;
    sun.shadow.camera.right = 40;
    sun.shadow.camera.top = 40;
    sun.shadow.camera.bottom = -40;
    sun.shadow.bias = -0.0001;
    sun.shadow.normalBias = 0.02;
    scene.add(sun);

    // 补光（减少暗部死黑）
    const fill = new THREE.DirectionalLight(0x4466aa, 0.3);
    fill.position.set(-5, 3, -5);
    scene.add(fill);
}

function addStarfield() {
    const count = 5000;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        pos[i*3] = (Math.random() - 0.5) * 200;
        pos[i*3+1] = Math.random() * 60 + 1;
        pos[i*3+2] = (Math.random() - 0.5) * 200;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
        color: 0x6688cc,
        size: 0.06,
        transparent: true,
        opacity: 0.35,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });
    scene.add(new THREE.Points(geo, mat));
}

function addGrid() {
    // 主网格
    const grid = new THREE.GridHelper(200, 200, 0x334466, 0x1a1a2e);
    grid.position.y = -0.49;
    scene.add(grid);

    // 高亮内圈网格（10m 间距）
    const inner = new THREE.GridHelper(200, 20, 0x5577aa, 0x1a1a2e);
    inner.position.y = -0.48;
    inner.material.opacity = 0.5;
    inner.material.transparent = true;
    scene.add(inner);

    // 坐标轴（彩色）
    const origin = new THREE.Vector3(0, -0.47, 0);
    const axisLen = 50;
    const makeAxis = (dir, color) => {
        const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 });
        const pts = [origin.clone(), origin.clone().add(dir.clone().multiplyScalar(axisLen))];
        scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
    };
    makeAxis(new THREE.Vector3(1, 0, 0), 0xff3333);
    makeAxis(new THREE.Vector3(0, 1, 0), 0x33ff33);
    makeAxis(new THREE.Vector3(0, 0, 1), 0x3388ff);
}

// ============================================================
// 轨迹渲染 — 发光管线
// ============================================================

function createGlowLine(points, color, opacity = 1) {
    if (points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(
        points.map(p => new THREE.Vector3(p[0], p[1], p[2]))
    );
    const samples = Math.max(120, points.length * 6);
    const curvePts = curve.getPoints(samples);

    const group = new THREE.Group();

    // 外层光晕（粗半透明）
    const glowGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
    const glowMat = new THREE.LineBasicMaterial({
        color,
        linewidth: 1,
        transparent: true,
        opacity: opacity * 0.25,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });
    group.add(new THREE.Line(glowGeo, glowMat));

    // 主线
    const lineGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
    const lineMat = new THREE.LineBasicMaterial({
        color,
        linewidth: 1,
        transparent: true,
        opacity: opacity * 0.85,
    });
    group.add(new THREE.Line(lineGeo, lineMat));

    return group;
}

function createPointSpheres(points, color, size = 0.07, opacity = 1) {
    const group = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.4,
        roughness: 0.25,
        metalness: 0.1,
        transparent: opacity < 1,
        opacity,
    });
    points.forEach(p => {
        const s = new THREE.Mesh(new THREE.SphereGeometry(size, 12, 12), mat);
        s.position.set(p[0], p[1], p[2]);
        s.castShadow = true;
        group.add(s);
    });
    return group;
}

function addLabel(text, pos, css) {
    const div = document.createElement('div');
    div.textContent = text;
    div.style.cssText = css;
    const label = new CSS2DObject(div);
    label.position.copy(new THREE.Vector3(pos[0], pos[1], pos[2]));
    return label;
}

function refreshAll() {
    // 清除旧元素
    for (const id in lines) removeTrail(id);
    for (const id in predLines) removePredTrail(id);
    for (const id in movingSpheres) { scene.remove(movingSpheres[id]); delete movingSpheres[id]; }
    for (const id in predictedSpheres) { scene.remove(predictedSpheres[id]); delete predictedSpheres[id]; }

    for (const [id, data] of Object.entries(detectionMethods)) {
        if (!data.visible || !data.points || data.points.length < 2) continue;

        // 发光轨迹
        const glow = createGlowLine(data.points, data.color);
        const spheres = createPointSpheres(data.points, data.color, 0.07);

        lines[id] = { glow, spheres };

        if (glow) scene.add(glow);
        scene.add(spheres);

        // 标签
        const pts = data.points;
        const startLab = addLabel('起点', pts[0],
            `color:#fff;font-size:10px;font-weight:600;background:rgba(0,0,0,0.65);padding:2px 8px;border-radius:10px;backdrop-filter:blur(8px);`);
        const endLab = addLabel('终点', pts[pts.length-1],
            `color:#fff;font-size:10px;font-weight:600;background:rgba(0,0,0,0.65);padding:2px 8px;border-radius:10px;backdrop-filter:blur(8px);`);
        const nameLab = addLabel(data.name, pts[Math.floor(pts.length/2)],
            `color:${data.color};font-size:12px;font-weight:700;background:rgba(0,0,0,0.7);padding:4px 12px;border-radius:14px;backdrop-filter:blur(8px);border:1px solid ${data.color}44;`);
        startLab.position.y += 0.35;
        endLab.position.y += 0.35;
        nameLab.position.y += 0.5;

        lines[id].startLabel = startLab;
        lines[id].endLabel = endLab;
        lines[id].nameLabel = nameLab;
        scene.add(startLab); scene.add(endLab); scene.add(nameLab);

        // 预测虚线
        const pred = predictedTrajectories[id];
        if (pred?.points?.length >= 2) {
            const predPts = pred.points.map(p => new THREE.Vector3(p[0],p[1],p[2]));
            const pGeo = new THREE.BufferGeometry().setFromPoints(predPts);
            const pLine = new THREE.Line(pGeo, new THREE.LineDashedMaterial({
                color: data.color, dashSize: 0.35, gapSize: 0.2,
                transparent: true, opacity: 0.55,
            }));
            pLine.computeLineDistances();
            const pGrp = createPointSpheres(pred.points, data.color, 0.05, 0.5);
            scene.add(pLine); scene.add(pGrp);
            predLines[id] = { line: pLine, points: pGrp };
        }
    }
}

function removeTrail(id) {
    const obj = lines[id];
    if (!obj) return;
    if (obj.glow) scene.remove(obj.glow);
    if (obj.spheres) scene.remove(obj.spheres);
    if (obj.startLabel) scene.remove(obj.startLabel);
    if (obj.endLabel) scene.remove(obj.endLabel);
    if (obj.nameLabel) scene.remove(obj.nameLabel);
    delete lines[id];
}

function removePredTrail(id) {
    const obj = predLines[id];
    if (!obj) return;
    if (obj.line) scene.remove(obj.line);
    if (obj.points) scene.remove(obj.points);
    delete predLines[id];
}

// ============================================================
// 拖尾粒子系统
// ============================================================

function ensureTrail(methodId, color) {
    if (trailSystems[methodId]) return;
    const count = 2000;
    const geo = new THREE.BufferGeometry();
    const posArr = new Float32Array(count * 3).fill(0);
    const colArr = new Float32Array(count * 3).fill(0);
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
    const mat = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        transparent: true,
        opacity: 0.7,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });
    const sys = new THREE.Points(geo, mat);
    scene.add(sys);

    const particles = [];
    for (let i = 0; i < count; i++) {
        particles.push({ pos: new THREE.Vector3(), life: 0, color: new THREE.Color(color) });
    }
    trailSystems[methodId] = { sys, particles, next: 0 };
}

function emitParticle(methodId, pos, color) {
    ensureTrail(methodId, color);
    const tp = trailSystems[methodId];
    const p = tp.particles[tp.next];
    p.pos.copy(pos);
    p.life = 1.0;
    p.color.set(color);
    tp.next = (tp.next + 1) % tp.particles.length;
}

function updateAllTrails() {
    for (const id in trailSystems) {
        const tp = trailSystems[id];
        const pArr = tp.sys.geometry.attributes.position.array;
        const cArr = tp.sys.geometry.attributes.color.array;
        for (let i = 0; i < tp.particles.length; i++) {
            const p = tp.particles[i];
            if (p.life > 0) {
                p.life -= 0.01;
                pArr[i*3] = p.pos.x; pArr[i*3+1] = p.pos.y; pArr[i*3+2] = p.pos.z;
                cArr[i*3] = p.color.r * p.life;
                cArr[i*3+1] = p.color.g * p.life;
                cArr[i*3+2] = p.color.b * p.life;
            } else {
                pArr[i*3] = pArr[i*3+1] = pArr[i*3+2] = -1000;
                cArr[i*3] = cArr[i*3+1] = cArr[i*3+2] = 0;
            }
        }
        tp.sys.geometry.attributes.position.needsUpdate = true;
        tp.sys.geometry.attributes.color.needsUpdate = true;
    }
}

// ============================================================
// 时间轴动画
// ============================================================

function lerp(pts, times, t) {
    if (!pts?.length || !times?.length) return null;
    if (t <= times[0]) return [...pts[0]];
    if (t >= times[times.length-1]) return [...pts[times.length-1]];
    let i = 0;
    while (i < times.length-1 && times[i+1] < t) i++;
    const r = (t - times[i]) / (times[i+1] - times[i]);
    const a = pts[i], b = pts[i+1];
    return [a[0]+(b[0]-a[0])*r, a[1]+(b[1]-a[1])*r, a[2]+(b[2]-a[2])*r];
}

function calcTimeRange() {
    let min = Infinity, max = -Infinity;
    for (const id in detectionMethods) {
        const ts = detectionMethods[id]?.timestamps;
        if (ts?.length) { min = Math.min(min, ts[0]); max = Math.max(max, ts[ts.length-1]); }
        const pred = predictedTrajectories[id];
        if (pred?.times?.length) max = Math.max(max, pred.times[pred.times.length-1]);
    }
    timeRange = { start: min === Infinity ? 0 : min, end: max === -Infinity ? 0 : max };
    return timeRange;
}

function setPredVisible(v) {
    for (const id in predLines) {
        if (predLines[id]?.line) predLines[id].line.visible = v;
        if (predLines[id]?.points) predLines[id].points.visible = v;
    }
}

function animStep(ts) {
    for (const id in detectionMethods) {
        const data = detectionMethods[id];
        const { points, timestamps } = data;
        const pred = predictedTrajectories[id];

        if (points?.length && timestamps?.length && ts <= timestamps[timestamps.length-1]) {
            const pos = lerp(points, timestamps, ts);
            if (pos) {
                if (!movingSpheres[id]) {
                    movingSpheres[id] = createGlowSphere(data.color, 0.18);
                    scene.add(movingSpheres[id]);
                }
                movingSpheres[id].position.set(pos[0], pos[1], pos[2]);
                movingSpheres[id].visible = true;
                emitParticle(id, movingSpheres[id].position, data.color);
            }
            if (predictedSpheres[id]) predictedSpheres[id].visible = false;
        } else if (pred?.points?.length && pred?.times?.length && ts <= pred.times[pred.times.length-1]) {
            const pos = lerp(pred.points, pred.times, ts);
            if (pos) {
                if (!predictedSpheres[id]) {
                    predictedSpheres[id] = createGlowSphere(data.color, 0.12, 0.7);
                    scene.add(predictedSpheres[id]);
                }
                predictedSpheres[id].position.set(pos[0], pos[1], pos[2]);
                predictedSpheres[id].visible = true;
                emitParticle(id, predictedSpheres[id].position, data.color);
            }
            if (movingSpheres[id]) movingSpheres[id].visible = false;
        }
    }
}

function createGlowSphere(color, size = 0.18, opacity = 1) {
    const group = new THREE.Group();
    // 内球
    const inner = new THREE.Mesh(
        new THREE.SphereGeometry(size, 20, 20),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.8, roughness: 0.2, transparent: opacity < 1, opacity })
    );
    inner.castShadow = true;
    group.add(inner);
    // 外光晕
    const glow = new THREE.Mesh(
        new THREE.SphereGeometry(size * 2.2, 16, 16),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: opacity * 0.2, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    group.add(glow);
    return group;
}

function startAnim() {
    if (animActive) return;
    calcTimeRange();
    if (timeRange.end <= timeRange.start) { toast.warning('无时间数据'); return; }
    animActive = true;
    animStart = performance.now();
    setPredVisible(false);

    const step = now => {
        if (!animActive) return;
        const ts = timeRange.start + (now - animStart) / 1000 * animSpeed;
        if (ts >= timeRange.end) { stopAnim(true); animStep(timeRange.end); return; }
        animStep(ts);
        document.getElementById('timelineSlider').value = ((ts - timeRange.start) / (timeRange.end - timeRange.start)) * 100;
        animId = requestAnimationFrame(step);
    };
    animId = requestAnimationFrame(step);
}

function stopAnim(restore = true) {
    if (animId) cancelAnimationFrame(animId);
    animActive = false; animId = null;
    if (restore) { setPredVisible(true); refreshAll(); }
}

// ============================================================
// UI 更新
// ============================================================

function updateRealtimeStats() {
    let total = 0;
    for (const id in detectionMethods) total += (detectionMethods[id].points || []).length;
    document.getElementById('totalPoints').textContent = total;

    for (const [id, data] of Object.entries(detectionMethods)) {
        if (!data.visible || (data.points || []).length < 2) continue;
        const pts = data.points;
        const ts = data.timestamps || [];
        const l = pts.length;
        const dt = ts[l-1] - ts[l-2] || 0.5;
        const speed = Math.sqrt((pts[l-1][0]-pts[l-2][0])**2 + (pts[l-1][2]-pts[l-2][2])**2) / dt;
        document.getElementById('currentSpeed').textContent = speed.toFixed(1) + ' m/s';
        document.getElementById('currentHeight').textContent = pts[l-1][1].toFixed(1) + ' m';
        break;
    }
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString('zh-CN');
}

window.toggleMethod = function(id) {
    detectionMethods[id].visible = !detectionMethods[id].visible;
    const badge = document.getElementById('status-' + id);
    if (badge) {
        badge.className = 'badge ' + (detectionMethods[id].visible ? 'badge-green' : 'badge-red');
        badge.textContent = detectionMethods[id].visible ? '显示' : '隐藏';
    }
    refreshAll();
    updateRealtimeStats();
};

// ============================================================
// 事件
// ============================================================

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onMouseMove(e) {
    mouse.x = (e.clientX / renderer.domElement.clientWidth) * 2 - 1;
    mouse.y = -(e.clientY / renderer.domElement.clientHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const targets = [];
    for (const id in lines) {
        if (lines[id]?.spheres) lines[id].spheres.children.forEach(s => targets.push(s));
    }
    const hits = raycaster.intersectObjects(targets);
    const tip = document.getElementById('coordTooltip');
    if (hits.length > 0) {
        const p = hits[0].point;
        tip.innerHTML = `📍 <span style="color:#0A84FF;">X</span>${p.x.toFixed(2)} <span style="color:#30D158;">Y</span>${p.y.toFixed(2)} <span style="color:#FF9F0A;">Z</span>${p.z.toFixed(2)}`;
        tip.style.display = 'block';
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top = (e.clientY - 24) + 'px';
    } else tip.style.display = 'none';
}

function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
}

function bindUI() {
    document.getElementById('playBtn').addEventListener('click', startAnim);
    document.getElementById('stopBtn').addEventListener('click', () => stopAnim(true));
    document.getElementById('speedSelect').addEventListener('change', e => { animSpeed = parseFloat(e.target.value); });
    document.getElementById('timelineSlider').addEventListener('input', e => {
        if (animActive) stopAnim(false);
        setPredVisible(false);
        calcTimeRange();
        const ts = timeRange.start + parseFloat(e.target.value) / 100 * (timeRange.end - timeRange.start);
        animStep(ts);
    });

    // 键盘：空格暂停/播放
    document.addEventListener('keydown', e => {
        if (e.code === 'Space' && e.target === document.body) {
            e.preventDefault();
            animActive ? stopAnim(true) : startAnim();
        }
    });
}

// ---- 启动 ----
init();
