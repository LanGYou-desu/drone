// 轨迹识别 — 主页面逻辑

async function init() {
    const statusPanel = document.getElementById('statusPanel');
    const dataPanel = document.getElementById('dataPanel');

    // 加载本模块状态
    try {
        const r = await (await fetch('/api/status')).json();
        statusPanel.innerHTML = `
            <div class="list-item">
                <span class="list-item-dot" style="background:var(--green);color:var(--green);"></span>
                <div class="list-item-text">
                    <div class="list-item-title">服务运行中</div>
                    <div class="list-item-sub">端口 5001 · v0.1.0</div>
                </div>
            </div>
            <div style="margin-top:8px;display:flex;gap:12px;">
                <span class="badge badge-blue">特征提取: ${r.features}</span>
                <span class="badge badge-blue">识别模型: ${r.models}</span>
                <span class="badge badge-blue">分类器: ${r.classifier}</span>
            </div>`;
    } catch {
        statusPanel.innerHTML = '<p style="color:var(--red);">状态加载失败</p>';
    }

    // 尝试读取共享数据
    try {
        const r = await (await fetch('http://127.0.0.1:5000/analysis/data')).json();
        const methods = Object.values(r);
        if (methods.length) {
            dataPanel.innerHTML = methods.map(m => `
                <div class="list-item">
                    <span class="list-item-dot" style="background:${m.color};color:${m.color};"></span>
                    <div class="list-item-text">
                        <div class="list-item-title">${m.name}</div>
                        <div class="list-item-sub">
                            高度点: ${m.heights?.length || 0} ·
                            速度样本: ${m.speeds?.length || 0} ·
                            加速度样本: ${m.accelerations?.length || 0}
                        </div>
                    </div>
                </div>
            `).join('');
        } else {
            dataPanel.innerHTML = '<p style="color:var(--text-tertiary);">暂无共享数据</p>';
        }
    } catch {
        dataPanel.innerHTML = '<p style="color:var(--orange);"><svg width="1em" height="1em" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:text-bottom;"><polygon points="8,2 15,14 1,14"/><line x1="8" y1="7" x2="8" y2="10"/><circle cx="8" cy="12" r="0.7" fill="currentColor"/></svg> 重建分析模块未运行（需端口 5000）</p>';
    }
}

init();
