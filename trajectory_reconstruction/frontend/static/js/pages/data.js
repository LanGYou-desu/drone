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
let selectedBackup = null;

async function loadBackupList() {
    const list = document.getElementById('backupList');
    list.innerHTML = '<p style="color:var(--text-secondary);"><span class="spinner"></span> 加载中...</p>';
    selectedBackup = null;
    try {
        const r = await (await fetch('/api/list_backups')).json();
        if (r.success && r.backups?.length) {
            list.innerHTML = '';
            r.backups.forEach(b => {
                const labelTag = b.label === 'auto' ? ' [自动]' : b.label === 'manual' ? ' [手动]' : '';
                const pts = b.point_count ? ` · ${b.point_count} 点` : '';

                const el = document.createElement('div');
                el.className = 'backup-item';
                el.innerHTML = `<div class="backup-item-name">${b.method}${labelTag}${pts}</div><div class="backup-item-date">${b.timestamp}</div>`;

                // 删除按钮
                const delBtn = document.createElement('button');
                delBtn.className = 'btn btn-danger btn-sm';
                delBtn.textContent = '✕';
                delBtn.title = '删除此备份';
                delBtn.style.cssText = 'position:absolute;right:8px;top:50%;transform:translateY(-50%);padding:2px 6px;font-size:0.7rem;';
                delBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    if (!confirm(`确认删除备份？\n${b.timestamp}`)) return;
                    try {
                        const dr = await (await fetch('/api/backup/delete', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ backup_name: b.filename }),
                        })).json();
                        dr.success ? toast.success(dr.message) : toast.error(dr.error);
                        if (dr.success) loadBackupList();
                    } catch (err) { toast.error(err.message); }
                });
                el.appendChild(delBtn);

                el.addEventListener('click', (ev) => {
                    if (ev.target.tagName === 'BUTTON') return;
                    list.querySelectorAll('.backup-item').forEach(x => x.classList.remove('selected'));
                    el.classList.add('selected');
                    selectedBackup = b.filename;
                });
                list.appendChild(el);
            });
        } else { list.innerHTML = '<p style="color:var(--text-secondary);">暂无备份</p>'; }
    } catch (e) { list.innerHTML = `<p style="color:var(--red);">加载失败</p>`; }
}

document.getElementById('listBackupsBtn').addEventListener('click', async () => {
    document.getElementById('backupModal').style.display = 'flex';
    loadBackupList();
});

document.getElementById('closeModalBtn').addEventListener('click', () => {
    document.getElementById('backupModal').style.display = 'none';
    selectedBackup = null;
});

document.getElementById('restoreSelectedBtn').addEventListener('click', async () => {
    if (!selectedBackup) { toast.warning('请先选择一个备份'); return; }
    try {
        const r = await (await fetch('/api/restore_backup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backup_file: selectedBackup }),
        })).json();
        r.success ? toast.success(r.message) : toast.error(r.error);
        if (r.success) setTimeout(() => location.reload(), 800);
    } catch (e) { toast.error(e.message); }
});

// ---- 一键恢复 ----
document.getElementById('restoreAllBtn').addEventListener('click', async () => {
    if (!confirm('一键恢复将覆盖所有当前数据，确认？')) return;
    try {
        const r = await (await fetch('/api/restore_all_backups', { method: 'POST' })).json();
        r.success ? toast.success(r.message) : toast.error(r.error);
        if (r.success) setTimeout(() => location.reload(), 800);
    } catch (e) { toast.error(e.message); }
});
