/**
 * 无人机检测 — 主页面控制器
 *
 * 管理双目视频源选择、平台切换、检测生命周期、UI 状态同步。
 * 依赖 detection.js 中的 DetectionAPI 模块。
 */

import { DetectionAPI } from './detection.js';

// ── 平台名称映射 ────────────────────────────────────

const PLATFORM_NAMES = {
    visible: '可见光', infrared: '红外', radar: '雷达', self: '自选',
};

// ── 双目视频源 ──────────────────────────────────────

const sources = {
    A: { dropZone: 'dropZoneA', fileInput: 'fileInputA', streamUrl: 'streamUrlA', fileInfo: 'fileInfoA', source: null },
    B: { dropZone: 'dropZoneB', fileInput: 'fileInputB', streamUrl: 'streamUrlB', fileInfo: 'fileInfoB', source: null },
};

Object.entries(sources).forEach(([ch, cfg]) => {
    const dz = document.getElementById(cfg.dropZone);
    const fi = document.getElementById(cfg.fileInput);
    const su = document.getElementById(cfg.streamUrl);

    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', (e) => {
        e.preventDefault();
        dz.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFile(ch, e.dataTransfer.files[0]);
    });

    fi.addEventListener('change', () => {
        if (fi.files.length > 0) handleFile(ch, fi.files[0]);
    });

    su.addEventListener('input', () => {
        if (su.value.trim()) {
            cfg.source = { type: 'stream', value: su.value.trim() };
            document.getElementById(cfg.fileInfo).style.display = 'none';
            updateStartButton();
        } else if (cfg.source && cfg.source.type === 'stream') {
            cfg.source = null;
            updateStartButton();
        }
    });
});

function handleFile(ch, file) {
    const validExt = /\.(mp4|avi|mov|mkv|webm)$/i;
    if (!validExt.test(file.name)) {
        window.toast?.warning(`${ch} 目: 不支持的格式，请使用 mp4/avi/mov/mkv`);
        return;
    }
    const cfg = sources[ch];
    cfg.source = { type: 'file', value: file };
    document.getElementById(cfg.fileInfo).style.display = 'block';
    document.getElementById(cfg.fileInfo).innerHTML =
        `<span style="color:var(--green);">✓</span> ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
    document.getElementById(cfg.streamUrl).value = '';
    updateStartButton();
}

function updateStartButton() {
    document.getElementById('btnStart').disabled = !(sources.A.source && sources.B.source);
}

// ── 平台选择 ──────────────────────────────────────

const platformSelect = document.getElementById('platformSelect');
platformSelect.addEventListener('change', () => {
    document.getElementById('detPlatform').textContent =
        PLATFORM_NAMES[platformSelect.value] || platformSelect.value;
});

// ── 检测控制 ──────────────────────────────────────

const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceVal = document.getElementById('confidenceVal');
confidenceSlider.addEventListener('input', () => {
    confidenceVal.textContent = parseFloat(confidenceSlider.value).toFixed(2);
});

let pollTimer = null;

const DetectionControl = {
    async start() {
        if (!sources.A.source || !sources.B.source) return;

        const config = {
            platform_id: platformSelect.value,
            confidence: parseFloat(confidenceSlider.value),
            frame_interval: parseInt(document.getElementById('frameInterval').value),
        };

        try {
            const result = await DetectionAPI.start(sources.A.source, sources.B.source, config);
            if (result.success) {
                this._setRunning(true);
                this._startPolling();
                window.toast?.success('双目检测已启动');
            }
        } catch (err) {
            window.toast?.error('启动失败: ' + err.message);
        }
    },

    async pause() { /* 同前 */ },
    async stop() { /* 同前 */ },
    async saveResults() { /* 同前 */ },

    // ... 其余内部方法与之前相同 ...
    async pause() {
        try { await DetectionAPI.pause(); this._setRunning(false); window.toast?.show('已暂停'); }
        catch { window.toast?.error('暂停失败'); }
    },

    async stop() {
        try {
            await DetectionAPI.stop();
            this._setRunning(false); this._stopPolling();
            window.toast?.show('检测已停止');
            this._showResults();
        } catch { window.toast?.error('停止失败'); }
    },

    async saveResults() {
        const btn = document.getElementById('btnSave');
        try {
            btn.disabled = true;
            btn.textContent = '已保存';
            const result = await DetectionAPI.saveResults();
            if (result.success) {
                window.toast?.success(`已保存 ${result.track_count} 条轨迹到 ${result.platform} 平台`);
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-secondary');
            } else {
                btn.disabled = false;
                btn.textContent = '保存到 data/';
                window.toast?.error('保存失败');
            }
        } catch (err) {
            btn.disabled = false;
            btn.textContent = '保存到 data/';
            window.toast?.error('保存失败: ' + err.message);
        }
    },

    _setRunning(running) {
        const btnStart = document.getElementById('btnStart');
        const btnPause = document.getElementById('btnPause');
        const btnStop = document.getElementById('btnStop');
        btnStart.style.display = running ? 'none' : '';
        btnPause.style.display = running ? '' : 'none';
        btnStart.disabled = running;
        btnStop.disabled = !running;
        document.querySelectorAll('.drop-zone').forEach(el => el.style.pointerEvents = running ? 'none' : '');
        document.querySelectorAll('input[type="text"]').forEach(el => el.disabled = running);
        document.querySelectorAll('input[type="file"]').forEach(el => el.disabled = running);
    },

    _startPolling() { this._stopPolling(); pollTimer = setInterval(() => this._poll(), 500); },
    _stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } },

    async _poll() {
        try {
            const status = await DetectionAPI.status();
            if (!status.session) return;
            const s = status.session;

            document.getElementById('detStatus').textContent = this._statusText(s.status);
            document.getElementById('detProgress').textContent = Math.round(s.progress * 100) + '%';
            document.getElementById('progressFill').style.width = (s.progress * 100) + '%';
            document.getElementById('detFrames').textContent = `${s.current_frame_a || 0} / ${s.total_frames}`;
            document.getElementById('detTrackCount').textContent = s.track_count;
            document.getElementById('det3DPoints').textContent = s.points_3d || 0;
            document.getElementById('detDuration').textContent = s.duration + 's';

            // 刷新预览帧（running/paused/completed 都刷新）
            if (s.status === 'running' || s.status === 'paused' || s.status === 'completed') {
                const ts = Date.now();
                const imgA = document.getElementById('imgA');
                const imgB = document.getElementById('imgB');
                if (imgA) imgA.src = DetectionAPI.previewUrl('a') + '&_=' + ts;
                if (imgB) imgB.src = DetectionAPI.previewUrl('b') + '&_=' + ts;
            }

            if (s.track_count > 0) this._updateTrackList();

            if (s.status === 'completed' || s.status === 'error') {
                this._stopPolling(); this._setRunning(false);
                if (s.status === 'error') {
                    document.getElementById('detStatus').textContent = '异常';
                    document.getElementById('resultCard').style.display = 'block';
                    document.getElementById('resultInfo').innerHTML =
                        '<span style=\"color:var(--red);\">检测失败: ' + (s.error || '未知错误，请检查终端日志') + '</span>';
                } else {
                    window.toast?.success('检测完成');
                    this._showResults();
                }
            }
        } catch { /* 静默 */ }
    },

    async _updateTrackList() {
        try {
            const data = await DetectionAPI.tracks('summary');
            if (!data.tracks) return;
            document.getElementById('trackList').innerHTML = data.tracks.map(t => `
                <div class="list-item">
                    <span class="list-item-dot" style="background:var(--blue);"></span>
                    <div class="list-item-text">
                        <div class="list-item-title">Track #${t.track_id} — ${t.class_name}</div>
                        <div class="list-item-sub">
                            ${t.point_count} 3D pts · 置信度 ${(t.confidence_avg*100).toFixed(0)}%
                            ${t.is_active ? '· <span style="color:var(--green);">活跃</span>' : ''}
                        </div>
                    </div>
                </div>
            `).join('');
        } catch { /* 静默 */ }
    },

    _showResults() {
        const card = document.getElementById('resultCard');
        card.style.display = 'block';
        const count = document.getElementById('detTrackCount').textContent;
        const plat = document.getElementById('detPlatform').textContent;
        document.getElementById('resultInfo').textContent =
            `双目检测完成，共 ${count} 个目标已定位 3D 坐标，可保存到「${plat}」平台供 3D 可视化使用。`;
    },

    _statusText(s) {
        const map = { idle:'待命中', running:'检测中', paused:'已暂停', completed:'已完成', error:'异常' };
        return map[s] || s;
    },
};

window.DetectionControl = DetectionControl;

// ── 页面加载时恢复状态 ──────────────────────────

async function restoreSession() {
    try {
        const data = await DetectionAPI.status();
        if (!data.session || data.session.status === 'idle') return;

        const s = data.session;
        DetectionControl._setRunning(s.status === 'running');
        document.getElementById('detStatus').textContent = DetectionControl._statusText(s.status);
        document.getElementById('detPlatform').textContent =
            ({visible:'可见光',infrared:'红外',radar:'雷达',self:'自选'})[s.platform_id] || s.platform_id;
        document.getElementById('progressFill').style.width = (s.progress * 100) + '%';
        document.getElementById('detTrackCount').textContent = s.track_count;

        if (s.status === 'running' || s.status === 'paused') {
            DetectionControl._startPolling();
        }
    } catch { /* 无活跃会话 */ }
}

// 离开页面时提醒
window.addEventListener('beforeunload', async (e) => {
    try {
        const data = await DetectionAPI.status();
        if (data.session && (data.session.status === 'running' || data.session.status === 'paused')) {
            e.preventDefault();
            e.returnValue = '检测正在进行中，确定离开？';
        }
    } catch { }
});

document.addEventListener('DOMContentLoaded', () => {
    updateStartButton();
    restoreSession();
});
