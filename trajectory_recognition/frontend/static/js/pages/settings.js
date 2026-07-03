/**
 * 检测设置页面 — 逻辑
 *
 * 独立的非模块脚本，管理设置表单和主题切换。
 */

// ── 主题切换 ──────────────────────────────────────

function updateThemeUI() {
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    var isDark = theme === 'dark';
    var sw = document.getElementById('themeSwitch');
    var label = document.getElementById('themeLabel');
    var icon = document.getElementById('themeIcon');

    if (isDark) {
        sw.classList.add('active');
        label.textContent = '深色模式';
        icon.innerHTML = '<svg width="1.2em" height="1.2em" viewBox="0 0 16 16" fill="currentColor"><path d="M13.5 10.5A6 6 0 0 1 5.5 2.5 6 6 0 1 0 13.5 10.5z"/></svg>';
    } else {
        sw.classList.remove('active');
        label.textContent = '浅色模式';
        icon.innerHTML = '<svg width="1.2em" height="1.2em" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"><circle cx="8" cy="8" r="3.5"/><line x1="8" y1="1" x2="8" y2="3"/><line x1="8" y1="13" x2="8" y2="15"/><line x1="1" y1="8" x2="3" y2="8"/><line x1="13" y1="8" x2="15" y2="8"/></svg>';
    }
}

function toggleTheme() {
    var current = document.documentElement.getAttribute('data-theme') || 'dark';
    var next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeUI();
}

// ── 自动保存开关 ──────────────────────────────────

function toggleAutoSave() {
    var sw = document.getElementById('autoSaveSwitch');
    sw.classList.toggle('active');
}

// ── 滑块标签更新 ──────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
    updateThemeUI();

    var confSlider = document.getElementById('detConfidence');
    var confLabel = document.getElementById('confLabel');
    if (confSlider) {
        confSlider.addEventListener('input', function () {
            confLabel.textContent = parseFloat(this.value).toFixed(2);
        });
    }

    var nmsSlider = document.getElementById('detNms');
    var nmsLabel = document.getElementById('nmsLabel');
    if (nmsSlider) {
        nmsSlider.addEventListener('input', function () {
            nmsLabel.textContent = parseFloat(this.value).toFixed(2);
        });
    }

    // 加载已保存的配置
    loadCurrentConfig();
});

// ── 配置加载与保存 ────────────────────────────────

async function loadCurrentConfig() {
    try {
        var r = await fetch('/api/detection/config');
        var data = await r.json();
        if (!data.config) return;
        var c = data.config;

        setVal('detModel', c.model);
        setVal('detDevice', c.device);
        setVal('detInputW', c.input_width);
        setVal('detInputH', c.input_height);
        setVal('detFrameInterval', c.frame_interval);
        setVal('detTracker', c.tracker);

        if (c.confidence_threshold !== undefined) {
            document.getElementById('detConfidence').value = c.confidence_threshold;
            document.getElementById('confLabel').textContent = c.confidence_threshold.toFixed(2);
        }
        if (c.nms_threshold !== undefined) {
            document.getElementById('detNms').value = c.nms_threshold;
            document.getElementById('nmsLabel').textContent = c.nms_threshold.toFixed(2);
        }
        if (c.auto_save !== undefined) {
            var sw = document.getElementById('autoSaveSwitch');
            if (c.auto_save) sw.classList.add('active');
            else sw.classList.remove('active');
        }

        // stereo params
        setVal('stereoBaseline', c.stereo?.baseline);
        setVal('stereoFovH', c.stereo?.fov_horizontal);
        setVal('stereoFovV', c.stereo?.fov_vertical);
        setVal('stereoResW', c.stereo?.resolution_width);
        setVal('stereoResH', c.stereo?.resolution_height);
        setVal('stereoTilt', c.stereo?.tilt_angle);
        setVal('stereoConv', c.stereo?.convergence_angle);
    } catch (e) { /* 加载失败使用默认值 */ }
}

async function saveAllSettings() {
    var config = {
        model: document.getElementById('detModel').value,
        device: document.getElementById('detDevice').value,
        input_width: parseInt(document.getElementById('detInputW').value),
        input_height: parseInt(document.getElementById('detInputH').value),
        confidence_threshold: parseFloat(document.getElementById('detConfidence').value),
        nms_threshold: parseFloat(document.getElementById('detNms').value),
        frame_interval: parseInt(document.getElementById('detFrameInterval').value),
        tracker: document.getElementById('detTracker').value,
        auto_save: document.getElementById('autoSaveSwitch').classList.contains('active'),
        stereo: {
            baseline: parseFloat(document.getElementById('stereoBaseline').value),
            fov_horizontal: parseFloat(document.getElementById('stereoFovH').value),
            fov_vertical: parseFloat(document.getElementById('stereoFovV').value),
            resolution_width: parseInt(document.getElementById('stereoResW').value),
            resolution_height: parseInt(document.getElementById('stereoResH').value),
            tilt_angle: parseFloat(document.getElementById('stereoTilt').value),
            convergence_angle: parseFloat(document.getElementById('stereoConv').value),
        },
    };

    try {
        var r = await fetch('/api/detection/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        var data = await r.json();
        if (data.success) {
            alert('设置已保存');
        } else {
            alert('保存失败: ' + (data.error || '未知错误'));
        }
    } catch (err) {
        // 降级：存储到 localStorage
        localStorage.setItem('detection_config', JSON.stringify(config));
        alert('设置已保存到本地（API 暂未实现）');
    }
}

function setVal(id, value) {
    var el = document.getElementById(id);
    if (el && value !== undefined) el.value = value;
}
