/**
 * 检测设置页面 — 逻辑
 * 每个卡片独立保存，与重建模块风格一致。
 */

// ── 主题切换 ──────────────────────────────────────

function updateThemeUI() {
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    var isDark = theme === 'dark';
    var sw = document.getElementById('themeSwitch');
    var label = document.getElementById('themeLabel');
    var icon = document.getElementById('themeIcon');
    if (isDark) {
        sw.classList.add('active'); label.textContent = '深色模式';
        icon.innerHTML = '<svg width="1.2em" height="1.2em" viewBox="0 0 16 16" fill="currentColor"><path d="M13.5 10.5A6 6 0 0 1 5.5 2.5 6 6 0 1 0 13.5 10.5z"/></svg>';
    } else {
        sw.classList.remove('active'); label.textContent = '浅色模式';
        icon.innerHTML = '<svg width="1.2em" height="1.2em" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"><circle cx="8" cy="8" r="3.5"/><line x1="8" y1="1" x2="8" y2="3"/><line x1="8" y1="13" x2="8" y2="15"/></svg>';
    }
}
function toggleTheme() {
    var next = (document.documentElement.getAttribute('data-theme')||'dark') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeUI();
}

function toggleAutoSave() {
    document.getElementById('autoSaveSwitch').classList.toggle('active');
}

// ── 分区域保存 ────────────────────────────────────

/**
 * 按区域保存设置到 config.json。
 * @param {'model'|'detection'|'stereo'} section
 */
async function saveSection(section) {
    var config = {};

    if (section === 'model') {
        config.model = document.getElementById('detModel').value;
        config.device = document.getElementById('detDevice').value;
        config.input_width = parseInt(document.getElementById('detInputW').value);
        config.input_height = parseInt(document.getElementById('detInputH').value);
    }
    if (section === 'detection') {
        config.confidence_threshold = parseFloat(document.getElementById('detConfidence').value);
        config.nms_threshold = parseFloat(document.getElementById('detNms').value);
        config.frame_interval = parseInt(document.getElementById('detFrameInterval').value);
        config.tracker = document.getElementById('detTracker').value;
        config.auto_save = document.getElementById('autoSaveSwitch').classList.contains('active');
    }
    if (section === 'stereo') {
        config.stereo = {
            baseline: parseFloat(document.getElementById('stereoBaseline').value),
            fov_horizontal: parseFloat(document.getElementById('stereoFovH').value),
            fov_vertical: parseFloat(document.getElementById('stereoFovV').value),
            resolution_width: parseInt(document.getElementById('stereoResW').value),
            resolution_height: parseInt(document.getElementById('stereoResH').value),
            tilt_angle: parseFloat(document.getElementById('stereoTilt').value),
            convergence_angle: parseFloat(document.getElementById('stereoConv').value),
        };
    }

    try {
        var r = await fetch('/api/detection/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        var data = await r.json();
        if (data.success) {
            var names = { model: '模型设置', detection: '检测参数', stereo: '标定参数' };
            showToast(names[section] + ' 已保存');
        } else {
            showToast('保存失败: ' + (data.error || ''), true);
        }
    } catch (err) {
        showToast('保存失败，已存到本地', true);
        var local = JSON.parse(localStorage.getItem('detection_config') || '{}');
        Object.assign(local, config);
        localStorage.setItem('detection_config', JSON.stringify(local));
    }
}

// ── 加载配置 ──────────────────────────────────────

async function loadConfig() {
    try {
        var r = await fetch('/api/detection/config');
        var data = await r.json();
        var c = (data && data.config) ? data.config : {};
        var d = c.detection || {};
        var s = c.stereo || {};

        setVal('detModel', d.model);
        setVal('detDevice', d.device);
        setVal('detInputW', d.input_width);
        setVal('detInputH', d.input_height);
        setVal('detFrameInterval', d.frame_interval);
        setVal('detTracker', d.tracker);
        if (d.confidence_threshold !== undefined) {
            document.getElementById('detConfidence').value = d.confidence_threshold;
            document.getElementById('confLabel').textContent = d.confidence_threshold.toFixed(2);
        }
        if (d.nms_threshold !== undefined) {
            document.getElementById('detNms').value = d.nms_threshold;
            document.getElementById('nmsLabel').textContent = d.nms_threshold.toFixed(2);
        }
        var sw = document.getElementById('autoSaveSwitch');
        if (d.auto_save !== false) sw.classList.add('active');
        else sw.classList.remove('active');

        setVal('stereoBaseline', s.baseline);
        setVal('stereoFovH', s.fov_horizontal);
        setVal('stereoFovV', s.fov_vertical);
        setVal('stereoResW', s.resolution_width);
        setVal('stereoResH', s.resolution_height);
        setVal('stereoTilt', s.tilt_angle);
        setVal('stereoConv', s.convergence_angle);
    } catch (e) { /* 默认值 */ }
}

function setVal(id, value) {
    if (value !== undefined && value !== null) {
        var el = document.getElementById(id);
        if (el) el.value = value;
    }
}

// ── Toast ──────────────────────────────────────────

function showToast(msg, isError) {
    var container = document.getElementById('toastContainer');
    if (!container) { alert(msg); return; }
    var el = document.createElement('div');
    el.className = 'toast' + (isError ? ' toast-error' : '');
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(function () { el.remove(); }, 2500);
}

// ── 初始化 ────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
    updateThemeUI();
    loadConfig();

    var cs = document.getElementById('detConfidence');
    if (cs) cs.addEventListener('input', function () {
        document.getElementById('confLabel').textContent = parseFloat(this.value).toFixed(2);
    });
    var ns = document.getElementById('detNms');
    if (ns) ns.addEventListener('input', function () {
        document.getElementById('nmsLabel').textContent = parseFloat(this.value).toFixed(2);
    });
});
