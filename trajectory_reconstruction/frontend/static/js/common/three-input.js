// Three.js 共享输入工具 — WASD 平移 + 鼠标悬停坐标拾取
import * as THREE from 'three';

/** 绑定 WASD/QE 键盘控制摄像头位置平移 */
export function bindWASD(camera, controls, speed = 0.12) {
    const keys = {};
    document.addEventListener('keydown', e => { keys[e.code] = true; });
    document.addEventListener('keyup', e => { keys[e.code] = false; });
    function loop() {
        requestAnimationFrame(loop);
        const dir = new THREE.Vector3();
        camera.getWorldDirection(dir);
        dir.normalize();
        const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
        const s = typeof speed === 'function' ? speed() : speed;
        if (keys['KeyW']) { camera.position.addScaledVector(dir, s); controls.target.addScaledVector(dir, s); }
        if (keys['KeyS']) { camera.position.addScaledVector(dir, -s); controls.target.addScaledVector(dir, -s); }
        if (keys['KeyA']) { camera.position.addScaledVector(right, -s); controls.target.addScaledVector(right, -s); }
        if (keys['KeyD']) { camera.position.addScaledVector(right, s); controls.target.addScaledVector(right, s); }
        if (keys['KeyQ']) { camera.position.y -= s; controls.target.y -= s; }
        if (keys['KeyE']) { camera.position.y += s; controls.target.y += s; }
    }
    loop();
}

/** 绑定鼠标悬停坐标拾取，显示 X/Y/Z/T tooltip */
export function bindCoordTooltip(renderer, camera, lines, predLines, tipId = 'coordTooltip') {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    raycaster.params.Points.threshold = 0.3;
    raycaster.params.Line = { threshold: 0.3 };
    let hoveredSphere = null;

    renderer.domElement.addEventListener('mousemove', e => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        const targets = [];
        for (const id in lines) {
            const spheres = lines[id]?.spheres || lines[id]?.points;
            if (spheres) {
                spheres.children.forEach(s => {
                    if (s.geometry?.type === 'SphereGeometry' && s.geometry.parameters.radius < 0.4) {
                        targets.push(s);
                    }
                });
            }
        }
        for (const id in predLines) {
            const spheres = predLines[id]?.points;
            if (spheres) {
                spheres.children.forEach(s => {
                    if (s.geometry?.type === 'SphereGeometry' && s.geometry.parameters.radius < 0.4) {
                        targets.push(s);
                    }
                });
            }
        }

        const hits = raycaster.intersectObjects(targets);
        const tip = document.getElementById(tipId);
        if (hoveredSphere && (!hits.length || hits[0].object !== hoveredSphere)) {
            hoveredSphere.scale.set(1, 1, 1);
            if (hoveredSphere.material.emissiveIntensity !== undefined) hoveredSphere.material.emissiveIntensity = 0.15;
            hoveredSphere = null;
        }

        if (hits.length > 0) {
            const obj = hits[0].object;
            const p = obj.position;
            if (obj !== hoveredSphere) {
                hoveredSphere = obj;
                hoveredSphere.scale.set(2.5, 2.5, 2.5);
                if (hoveredSphere.material.emissiveIntensity !== undefined) hoveredSphere.material.emissiveIntensity = 1.5;
            }
            if (tip) {
                let timeStr = '';
                const ud = obj.userData;
                if (ud?.ts && ud.idx != null && ud.idx < ud.ts.length) {
                    timeStr = ` <span style="color:#d29922">T</span>${ud.ts[ud.idx].toFixed(2)}`;
                }
                tip.innerHTML = `<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--blue);margin-right:4px;vertical-align:middle;"></span><span style="color:#f85149">X</span>${p.x.toFixed(2)} <span style="color:#3fb950">Y</span>${p.y.toFixed(2)} <span style="color:#58a6ff">Z</span>${p.z.toFixed(2)}${timeStr}`;
                tip.style.display = 'block';
                tip.style.left = (e.clientX + 16) + 'px';
                tip.style.top = (e.clientY - 28) + 'px';
            }
        } else {
            if (tip) tip.style.display = 'none';
        }
    });
}

/** 绑定窗口 resize */
export function bindResize(camera, renderer, labelRenderer) {
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
    });
}
