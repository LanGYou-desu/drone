/**
 * 检测设置页面 — 每平台独立保存立体参数和空间位置
 */

var _stereoData = {}, _posData = {};

// ── 主题 ──────────────────────────────────────────

function updateThemeUI() {
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    var isDark = theme === 'dark';
    var sw = document.getElementById('themeSwitch');
    var label = document.getElementById('themeLabel');
    var icon = document.getElementById('themeIcon');
    if (isDark) { sw.classList.add('active'); label.textContent = '深色模式';
        icon.innerHTML = '<svg width="1.2em" height="1.2em" viewBox="0 0 16 16" fill="currentColor"><path d="M13.5 10.5A6 6 0 0 1 5.5 2.5 6 6 0 1 0 13.5 10.5z"/></svg>';
    } else { sw.classList.remove('active'); label.textContent = '浅色模式';
        icon.innerHTML = '<svg width="1.2em" height="1.2em" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"><circle cx="8" cy="8" r="3.5"/></svg>';
    }
}
function toggleTheme() {
    var next = (document.documentElement.getAttribute('data-theme')||'dark') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next); localStorage.setItem('theme', next); updateThemeUI();
}
function toggleAutoSave() { document.getElementById('autoSaveSwitch').classList.toggle('active'); }

// ── 加载立体参数 ───────────────────────────────

function loadStereoPlatform() {
    var pid = document.getElementById('stereoPlatform').value;
    var d = (_stereoData && _stereoData[pid]) ? _stereoData[pid] : {};
    setVal('stereoBaseline', d.baseline, 1.0);
    setVal('stereoFovH', d.fov_horizontal, 90.0);
    setVal('stereoFovV', d.fov_vertical, 60.0);
    setVal('stereoResW', d.resolution_width, 1920);
    setVal('stereoResH', d.resolution_height, 1080);
    setVal('stereoTilt', d.tilt_angle, 0.0);
    setVal('stereoConv', d.convergence_angle, 0.0);
}

function loadPosPlatform() {
    var pid = document.getElementById('posPlatform').value;
    var d = (_posData && _posData[pid]) ? _posData[pid] : {};
    setVal('posX', d.x, 0.0);
    setVal('posY', d.y, 1.5);
    setVal('posZ', d.z, 0.0);
    setVal('posPitch', d.pitch, 0.0);
    setVal('posYaw', d.yaw, 0.0);
    setVal('posRoll', d.roll, 0.0);
}

// ── 分区域保存 ──────────────────────────────────

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
        var tcRaw = document.getElementById('detTargetClasses').value.trim();
        if (tcRaw) config.target_classes = tcRaw.split(',').map(function(s){return parseInt(s.trim());}).filter(function(n){return !isNaN(n);});
        else config.target_classes = null;
    }
    if (section === 'stereo') {
        var pid = document.getElementById('stereoPlatform').value;
        var d = {
            baseline: parseFloat(document.getElementById('stereoBaseline').value),
            fov_horizontal: parseFloat(document.getElementById('stereoFovH').value),
            fov_vertical: parseFloat(document.getElementById('stereoFovV').value),
            resolution_width: parseInt(document.getElementById('stereoResW').value),
            resolution_height: parseInt(document.getElementById('stereoResH').value),
            tilt_angle: parseFloat(document.getElementById('stereoTilt').value),
            convergence_angle: parseFloat(document.getElementById('stereoConv').value),
        };
        _stereoData[pid] = d;
        config.stereo = _stereoData;
    }
    if (section === 'position') {
        var pid = document.getElementById('posPlatform').value;
        var d = {
            x: parseFloat(document.getElementById('posX').value),
            y: parseFloat(document.getElementById('posY').value),
            z: parseFloat(document.getElementById('posZ').value),
            pitch: parseFloat(document.getElementById('posPitch').value),
            yaw: parseFloat(document.getElementById('posYaw').value),
            roll: parseFloat(document.getElementById('posRoll').value),
        };
        _posData[pid] = d;
        config.platform_positions = _posData;
    }

    try {
        var r = await fetch('/api/detection/config', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(config) });
        var data = await r.json();
        if (data.success) {
            var names = {model:'模型设置', detection:'检测参数', stereo:'立体参数', position:'空间位置'};
            showToast(names[section] + ' 已保存');
        } else { showToast('保存失败: ' + (data.error || ''), true); }
    } catch (err) { showToast('保存失败', true); }
}

// ── 加载配置 ─────────────────────────────────────

async function loadConfig() {
    try {
        var r = await fetch('/api/detection/config');
        var data = await r.json();
        var c = (data && data.config) ? data.config : {};
        var d = c.detection || {};
        setVal('detModel', d.model); setVal('detDevice', d.device);
        setVal('detInputW', d.input_width); setVal('detInputH', d.input_height);
        setVal('detFrameInterval', d.frame_interval); setVal('detTracker', d.tracker);
        if (d.target_classes && d.target_classes.length) setVal('detTargetClasses', d.target_classes.join(','));
        if (d.confidence_threshold != null) {
            document.getElementById('detConfidence').value = d.confidence_threshold;
            document.getElementById('confLabel').textContent = d.confidence_threshold.toFixed(2);
        }
        if (d.nms_threshold != null) {
            document.getElementById('detNms').value = d.nms_threshold;
            document.getElementById('nmsLabel').textContent = d.nms_threshold.toFixed(2);
        }
        var sw = document.getElementById('autoSaveSwitch');
        if (d.auto_save !== false) sw.classList.add('active'); else sw.classList.remove('active');

        _stereoData = c.stereo || {};
        _posData = c.platform_positions || {};
        loadStereoPlatform();
        loadPosPlatform();
    } catch (e) { /* default */ }
}

function setVal(id, value, def) {
    var el = document.getElementById(id);
    if (el && value != null) el.value = value;
    else if (el && def != null) el.value = def;
}

function showToast(msg, isError) {
    var ct = document.getElementById('toastContainer');
    if (!ct) { alert(msg); return; }
    var el = document.createElement('div');
    el.className = 'toast' + (isError ? ' toast-error' : '');
    el.textContent = msg; ct.appendChild(el);
    setTimeout(function(){el.remove();}, 2500);
}

// ── 初始化 ──────────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
    updateThemeUI(); loadConfig();
    var cs = document.getElementById('detConfidence');
    if (cs) cs.addEventListener('input', function(){document.getElementById('confLabel').textContent = parseFloat(this.value).toFixed(2);});
    var ns = document.getElementById('detNms');
    if (ns) ns.addEventListener('input', function(){document.getElementById('nmsLabel').textContent = parseFloat(this.value).toFixed(2);});
});
