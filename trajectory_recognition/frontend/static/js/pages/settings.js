/**
 * 检测设置页面
 */

var _platformsData = {};

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

// ── 加载平台参数（立体+位置）───────────────────

function loadPlatformParams() {
    var pid = document.getElementById('platSelect').value;
    var p = (_platformsData && _platformsData[pid]) ? _platformsData[pid] : {};
    setVal('stereoFocal', p.focal_length_px, 0);
    setVal('stereoBaseline', p.baseline, 1.0);
    setVal('stereoFovH', p.fov_horizontal, 90.0);
    setVal('stereoFovV', p.fov_vertical, 60.0);
    setVal('stereoResW', p.resolution_width, 1920);
    setVal('stereoResH', p.resolution_height, 1080);
    setVal('posX', p.pos_x, 0.0);
    setVal('posY', p.pos_y, 1.5);
    setVal('posZ', p.pos_z, 0.0);
    setVal('posPitch', p.pitch, 0.0);
    setVal('posYaw', p.yaw, 0.0);
    setVal('posRoll', p.roll, 0.0);
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
        var tc = document.getElementById('detTargetClasses').value.trim();
        if (tc) config.target_classes = tc.split(',').map(function(s){return parseInt(s.trim());}).filter(function(n){return !isNaN(n);});
        else config.target_classes = null;
    }
    if (section === 'platform') {
        var pid = document.getElementById('platSelect').value;
        _platformsData[pid] = {
            focal_length_px: parseFloat(document.getElementById('stereoFocal').value) || 0,
            baseline: parseFloat(document.getElementById('stereoBaseline').value),
            fov_horizontal: parseFloat(document.getElementById('stereoFovH').value),
            fov_vertical: parseFloat(document.getElementById('stereoFovV').value),
            resolution_width: parseInt(document.getElementById('stereoResW').value),
            resolution_height: parseInt(document.getElementById('stereoResH').value),
            pos_x: parseFloat(document.getElementById('posX').value),
            pos_y: parseFloat(document.getElementById('posY').value),
            pos_z: parseFloat(document.getElementById('posZ').value),
            pitch: parseFloat(document.getElementById('posPitch').value),
            yaw: parseFloat(document.getElementById('posYaw').value),
            roll: parseFloat(document.getElementById('posRoll').value),
        };
        config.platforms = _platformsData;
    }

    try {
        var r = await fetch('/api/detection/config', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(config) });
        var data = await r.json();
        if (data.success) {
            var names = {model:'模型设置', detection:'检测参数', platform:'平台参数'};
            showToast(names[section] + ' 已保存');
        } else { showToast('保存失败', true); }
    } catch (err) { showToast('保存失败', true); }
}

// ── 加载模型列表 ─────────────────────────────────

async function loadModelList() {
    try {
        var r = await fetch('/api/detection/models');
        var data = await r.json();
        var sel = document.getElementById('detModel');
        if (!data.success || !data.models || !data.models.length) {
            sel.innerHTML = '<option value="">无可用模型</option>';
            return;
        }
        sel.innerHTML = data.models.map(function(m) {
            return '<option value="' + m.path + '">' + m.name + ' (' + m.size_mb + 'MB)</option>';
        }).join('');
    } catch (e) {
        document.getElementById('detModel').innerHTML = '<option value="">加载失败</option>';
    }
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
        if (d.target_classes && d.target_classes.length) {
            setVal('detTargetClasses', d.target_classes.join(','));
        } else if (d.target_classes !== undefined) {
            setVal('detTargetClasses', '');  // null/空数组 → 清空输入框（检测全部）
        }
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

        _platformsData = c.platforms || {};
        loadPlatformParams();
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

document.addEventListener('DOMContentLoaded', async function () {
    updateThemeUI();
    await loadModelList();       // 先加载模型列表
    await loadConfig();           // 再加载配置（会选中当前模型）
    var cs = document.getElementById('detConfidence');
    if (cs) cs.addEventListener('input', function(){document.getElementById('confLabel').textContent = parseFloat(this.value).toFixed(2);});
    var ns = document.getElementById('detNms');
    if (ns) ns.addEventListener('input', function(){document.getElementById('nmsLabel').textContent = parseFloat(this.value).toFixed(2);});
});
