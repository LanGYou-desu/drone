// Three.js 共享 3D 工具 — buildAxes（自定义彩色坐标轴）· 支持明暗主题
import * as THREE from 'three';

function isLightTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light';
}

export function getSceneBackground() {
    return isLightTheme() ? 0x6b7077 : 0x0d0d18;
}

export function getFogColor() {
    return isLightTheme() ? 0x5f656c : 0x0d0d18;
}

export function getGridColor() {
    return isLightTheme() ? { main: 0x555a61, sub: 0x7a7f86 } : { main: 0x334466, sub: 0x1a1a2e };
}

export function buildAxes(scene, CSS2DObject) {
    const light = isLightTheme();
    const origin = new THREE.Vector3(0, -0.48, 0);
    const len = 15, tickStep = 2, tickSize = 0.2;
    // 浅色模式下使用更深的颜色以保证在浅背景上的可读性
    const colors = light
        ? { x: 0xcc2222, y: 0x22aa22, z: 0x2266dd }
        : { x: 0xff4444, y: 0x44ff44, z: 0x4488ff };
    const dirs = {
        x: new THREE.Vector3(1, 0, 0),
        y: new THREE.Vector3(0, 1, 0),
        z: new THREE.Vector3(0, 0, 1),
    };

    for (const [axis, dir] of Object.entries(dirs)) {
        const color = colors[axis];
        const end = origin.clone().add(dir.clone().multiplyScalar(len));

        // 正半轴
        scene.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([origin, end]),
            new THREE.LineBasicMaterial({ color })
        ));
        // 负半轴虚线
        const negEnd = origin.clone().add(dir.clone().multiplyScalar(-len * 0.3));
        scene.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([origin, negEnd]),
            new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: light ? 0.25 : 0.4 })
        ));
        // 刻度
        for (let t = tickStep; t <= len; t += tickStep) {
            const tc = origin.clone().add(dir.clone().multiplyScalar(t));
            const cross = axis === 'x' ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
            scene.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([
                    tc.clone().add(cross.clone().multiplyScalar(tickSize)),
                    tc.clone().add(cross.clone().multiplyScalar(-tickSize)),
                ]),
                new THREE.LineBasicMaterial({ color, transparent: true, opacity: light ? 0.45 : 0.5 })
            ));
            // CSS2D 刻度数字
            if (CSS2DObject) {
                const div = document.createElement('div');
                div.textContent = String(t);
                div.style.cssText = `color:#${color.toString(16).padStart(6,'0')};font-size:9px;font-weight:600;font-family:SF Mono,monospace;`;
                const label = new CSS2DObject(div);
                label.position.copy(tc); label.position.y -= 0.3;
                scene.add(label);
            }
        }
        // 箭头
        const arrow = new THREE.Mesh(
            new THREE.ConeGeometry(0.15, 0.5, 6),
            new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: light ? 0.3 : 0.5 })
        );
        arrow.position.copy(end);
        if (axis === 'x') arrow.rotation.z = -Math.PI / 2;
        else if (axis === 'z') arrow.rotation.x = Math.PI / 2;
        scene.add(arrow);
        // 轴字母
        if (CSS2DObject) {
            const ld = document.createElement('div');
            ld.textContent = axis.toUpperCase();
            ld.style.cssText = `color:#${color.toString(16).padStart(6,'0')};font-size:13px;font-weight:700;font-family:SF Pro Display,sans-serif;`;
            const ll = new CSS2DObject(ld);
            ll.position.copy(end.clone().add(dir.clone().multiplyScalar(0.7)));
            scene.add(ll);
        }
    }
}
