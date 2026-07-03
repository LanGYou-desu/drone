/**
 * 检测历史 — 页面逻辑
 *
 * 直接从 data/fact/ 和 data/backup/ 读取已保存的轨迹文件。
 * 数据存储即文件系统，无需额外的 session 记录。
 */

const RECON_BASE = 'http://127.0.0.1:5000';  // 仅用于"在 3D 视图中查看"跳转

const HistoryPage = {
    async refresh() {
        await Promise.all([
            this._loadDetectFiles(),
            this._loadBackups(),
        ]);
    },

    // ── data/fact/ 中的检测文件 ──

    async _loadDetectFiles() {
        const el = document.getElementById('dataFileList');
        try {
            const r = await fetch('/api/detection/files');
            const data = await r.json();
            if (!data.success || !data.files || data.files.length === 0) {
                el.innerHTML = '<p style="color:var(--text-tertiary);font-size:0.85rem;text-align:center;padding:24px;">暂无已保存的检测轨迹<br><small>完成检测后点击"保存到 data/"即可在此查看</small></p>';
                this._updateStats(0, 0);
                return;
            }

            const files = data.files;
            const totalPoints = files.reduce((s, f) => s + (f.point_count || 0), 0);

            el.innerHTML = files.map(f => `
                <div class="list-item" style="padding:12px;">
                    <span class="list-item-dot" style="background:var(--green);"></span>
                    <div class="list-item-text" style="flex:1;">
                        <div class="list-item-title">${f.name}</div>
                        <div class="list-item-sub">
                            ${f.point_count} 轨迹点 · ${f.size_kb} KB
                            ${f.modified ? ' · ' + new Date(f.modified).toLocaleString('zh-CN') : ''}
                        </div>
                    </div>
                    <div style="display:flex;gap:4px;">
                        <a class="btn btn-ghost btn-sm" href="${RECON_BASE}" title="在 3D 视图中查看">查看</a>
                        <button class="btn btn-ghost btn-sm" onclick="HistoryPage._deleteFile('${f.name}')" title="删除">
                            <svg width="1em" height="1em" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"><line x1="2" y1="3.5" x2="14" y2="3.5"/><rect x="3.5" y="4" width="9" height="10" rx="0.8"/><line x1="6.5" y1="6" x2="6.5" y2="12"/><line x1="9.5" y1="6" x2="9.5" y2="12"/></svg>
                        </button>
                    </div>
                </div>
            `).join('');

            this._updateStats(files.length, totalPoints);
        } catch (err) {
            el.innerHTML = '<p style="color:var(--orange);font-size:0.85rem;text-align:center;padding:24px;">无法读取 data/fact/，请检查服务状态</p>';
        }
    },

    async _deleteFile(name) {
        if (!confirm(`确定删除 ${name}？此操作不可恢复。`)) return;
        try {
            const r = await fetch(`/api/detection/files/${encodeURIComponent(name)}`, { method: 'DELETE' });
            const data = await r.json();
            if (data.success) {
                window.toast?.success(`已删除 ${name}`);
                this.refresh();
            } else {
                window.toast?.error('删除失败: ' + (data.error || ''));
            }
        } catch (err) {
            window.toast?.error('删除失败: ' + err.message);
        }
    },

    // ── data/backup/ 备份列表（本模块自给自足）──

    async _loadBackups() {
        const el = document.getElementById('backupList');
        try {
            const r = await fetch('/api/detection/backups');
            const data = await r.json();
            if (!data.success || !data.backups || data.backups.length === 0) {
                el.innerHTML = '<p style="color:var(--text-tertiary);font-size:0.85rem;text-align:center;padding:24px;">暂无备份</p>';
                document.getElementById('backupCount').textContent = '0';
                return;
            }

            document.getElementById('backupCount').textContent = data.backups.length;

            el.innerHTML = data.backups.map(b => {
                return `
                <div class="list-item" style="padding:12px;">
                    <span class="list-item-dot" style="background:var(--orange);"></span>
                    <div class="list-item-text" style="flex:1;">
                        <div class="list-item-title">
                            ${b.name}
                            <span class="badge badge-blue" style="margin-left:8px;">${b.label || 'auto'}</span>
                        </div>
                        <div class="list-item-sub">
                            ${b.file_count} 个文件 · ${b.total_points} 个轨迹点
                            ${b.timestamp ? ' · ' + new Date(b.timestamp).toLocaleString('zh-CN') : ''}
                        </div>
                    </div>
                </div>`;
            }).join('');
        } catch (err) {
            el.innerHTML = '<p style="color:var(--text-tertiary);font-size:0.85rem;text-align:center;padding:24px;">暂无备份数据</p>';
            document.getElementById('backupCount').textContent = '—';
        }
    },

    // ── 统计 ──

    _updateStats(fileCount, totalPoints) {
        document.getElementById('savedFiles').textContent = fileCount;
        document.getElementById('totalTracks').textContent = totalPoints;
    },
};

// 初始化
document.addEventListener('DOMContentLoaded', () => HistoryPage.refresh());
window.HistoryPage = HistoryPage;
