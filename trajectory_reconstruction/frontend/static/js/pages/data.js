/**
 * Data Page — 数据管理
 * 高内聚：上传/备份/恢复/清理全部在此模块
 * 低耦合：通过 API 与后端交互，刷新即同步
 */
import { toast } from '../common/toast.js';

// ---- 上传 ----
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const file = document.getElementById('dataFile').files[0];
    if (!file) { toast.warning('请先选择文件'); return; }
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true; btn.textContent = '⏳ 上传中...';
    try {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('method_id', 'self');
        const resp = await fetch('/api/load_data', { method: 'POST', body: fd });
        const r = await resp.json();
        r.success ? toast.success(`已加载 ${r.name}`) : toast.error(r.error);
        if (r.success) setTimeout(() => location.reload(), 800);
    } catch (e) { toast.error(e.message); }
    finally { btn.disabled = false; btn.textContent = '上传为自选平台'; }
});

// ---- 刷新 ----
document.getElementById('refreshBtn').addEventListener('click', async () => {
    const btn = document.getElementById('refreshBtn');
    btn.disabled = true;
    try {
        const r = await (await fetch('/api/refresh_data', { method: 'POST' })).json();
        r.success ? toast.success('已重置') : toast.error('重置失败');
        if (r.success) setTimeout(() => location.reload(), 800);
    } catch (e) { toast.error(e.message); }
    finally { btn.disabled = false; }
});

// ---- 清理 ----
document.getElementById('clearAllBtn').addEventListener('click', async () => {
    if (!confirm('确认清理所有数据？数据会自动备份到 backup/。')) return;
    const btn = document.getElementById('clearAllBtn');
    btn.disabled = true; btn.textContent = '⏳ 清理中...';
    try {
        const r = await (await fetch('/api/clear_all_data', { method: 'POST' })).json();
        r.success ? toast.success('已清理并备份') : toast.error('清理失败');
        if (r.success) setTimeout(() => location.reload(), 800);
    } catch (e) { toast.error(e.message); }
    finally { btn.disabled = false; btn.textContent = '🗑️ 清理全部数据'; }
});

// ---- 手动备份 ----
document.getElementById('createBackupBtn').addEventListener('click', async () => {
    const btn = document.getElementById('createBackupBtn');
    btn.disabled = true; btn.textContent = '⏳ 备份中...';
    try {
        const r = await (await fetch('/api/backup/create', { method: 'POST' })).json();
        r.success ? toast.success(r.message) : toast.error(r.error);
    } catch (e) { toast.error(e.message); }
    finally { btn.disabled = false; btn.textContent = '📸 立即备份'; }
});

// ---- 备份列表 ----
let selectedBackups = new Set();

async function loadBackupList() {
    const list = document.getElementById('backupList');
    list.innerHTML = '<p style="color:var(--text-secondary);"><span class="spinner"></span> 加载中...</p>';
    selectedBackups.clear();
    try {
        const r = await (await fetch('/api/list_backups')).json();
        if (r.success && r.backups?.length) {
            list.innerHTML = '';
            r.backups.forEach(b => {
                const labelTag = b.label === 'auto' ? '自动' : '手动';
                const pts = b.point_count ? `${b.point_count} 点` : '';

                const el = document.createElement('div');
                el.className = 'backup-item';
                el.style.cssText = 'display:flex;align-items:center;gap:10px;';

                // 复选框
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.style.cssText = 'flex-shrink:0;cursor:pointer;';
                cb.addEventListener('click', (e) => e.stopPropagation());
                cb.addEventListener('change', () => {
                    cb.checked ? selectedBackups.add(b.filename) : selectedBackups.delete(b.filename);
                    updateBackupFooter();
                });

                // 信息区
                const info = document.createElement('div');
                info.style.cssText = 'flex:1;min-width:0;';
                info.innerHTML = `<div class="backup-item-name">${b.method} · ${pts}</div><div class="backup-item-date">${b.timestamp} &nbsp;<span style="font-size:0.7rem;color:var(--text-secondary);">[${labelTag}]</span></div>`;

                // 点击行切换选中
                el.addEventListener('click', (ev) => {
                    if (ev.target.tagName === 'INPUT') return;
                    cb.checked = !cb.checked;
                    cb.dispatchEvent(new Event('change'));
                });

                el.appendChild(cb);
                el.appendChild(info);
                list.appendChild(el);
            });
        } else { list.innerHTML = '<p style="color:var(--text-secondary);">暂无备份</p>'; }
    } catch (e) { list.innerHTML = `<p style="color:var(--red);">加载失败</p>`; }
    updateBackupFooter();
}

function updateBackupFooter() {
    const btn = document.getElementById('deleteSelectedBtn');
    const restoreBtn = document.getElementById('restoreSelectedBtn');
    if (btn) btn.disabled = selectedBackups.size === 0;
    if (restoreBtn) restoreBtn.disabled = selectedBackups.size !== 1;
}

// 全选
document.getElementById('selectAllBtn')?.addEventListener('click', () => {
    const list = document.getElementById('backupList');
    const cbs = list.querySelectorAll('input[type="checkbox"]');
    const allChecked = Array.from(cbs).every(c => c.checked);
    cbs.forEach(cb => { cb.checked = !allChecked; cb.dispatchEvent(new Event('change')); });
});

document.getElementById('listBackupsBtn').addEventListener('click', async () => {
    document.getElementById('backupModal').style.display = 'flex';
    loadBackupList();
});

document.getElementById('closeModalBtn').addEventListener('click', () => {
    document.getElementById('backupModal').style.display = 'none';
    selectedBackups.clear();
});

// 恢复选中（单个或批量恢复最新）
document.getElementById('restoreSelectedBtn').addEventListener('click', async () => {
    if (selectedBackups.size === 0) { toast.warning('请先勾选备份'); return; }
    const name = [...selectedBackups][0];
    try {
        const r = await (await fetch('/api/restore_backup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backup_file: name }),
        })).json();
        r.success ? toast.success(r.message) : toast.error(r.error);
        if (r.success) setTimeout(() => location.reload(), 800);
    } catch (e) { toast.error(e.message); }
});

// 批量删除选中
async function deleteSelected() {
    if (selectedBackups.size === 0) return;
    if (!confirm(`确认删除 ${selectedBackups.size} 个备份？`)) return;
    let ok = 0, fail = 0;
    for (const name of selectedBackups) {
        try {
            const r = await (await fetch('/api/backup/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ backup_name: name }),
            })).json();
            r.success ? ok++ : fail++;
        } catch { fail++; }
    }
    if (ok > 0) toast.success(`已删除 ${ok} 个备份` + (fail ? `，${fail} 个失败` : ''));
    else toast.error('删除失败');
    loadBackupList();
}

document.getElementById('deleteSelectedBtn').addEventListener('click', deleteSelected);

// ---- 一键恢复 ----
document.getElementById('restoreAllBtn').addEventListener('click', async () => {
    if (!confirm('一键恢复将覆盖所有当前数据，确认？')) return;
    try {
        const r = await (await fetch('/api/restore_all_backups', { method: 'POST' })).json();
        r.success ? toast.success(r.message) : toast.error(r.error);
        if (r.success) setTimeout(() => location.reload(), 800);
    } catch (e) { toast.error(e.message); }
});
