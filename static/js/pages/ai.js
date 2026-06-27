/**
 * AI Page — 大模型捕捉策略生成
 * 高内聚：AI 对话逻辑封装在此模块
 * 低耦合：仅通过 POST /api/ai_suggestion 与后端通信
 */
import { toast } from '../common/toast.js';

const { methodsData } = window._PAGE_DATA_ || {};

async function generate() {
    const btn = document.getElementById('aiBtn');
    const output = document.getElementById('aiOutput');
    btn.disabled = true;
    btn.textContent = '⏳ AI 分析中...';
    output.innerHTML = '<p style="color:var(--text-secondary);"><span class="spinner"></span> 正在调用大模型分析多平台轨迹数据…</p>';

    // 收集选中的平台
    const checks = document.querySelectorAll('.ai-platform-check:checked');
    const selected = {};
    checks.forEach(cb => {
        const id = cb.value;
        if (methodsData[id]) {
            selected[id] = {
                name: methodsData[id].name,
                points: methodsData[id].points || [],
                timestamps: methodsData[id].timestamps || [],
            };
        }
    });

    if (Object.keys(selected).length === 0) {
        toast.warning('请至少选择一个平台');
        btn.disabled = false;
        btn.textContent = '💡 生成捕捉策略';
        return;
    }

    try {
        const resp = await fetch('/api/ai_suggestion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ methods_data: selected }),
        });
        const result = await resp.json();
        if (result.success) {
            output.innerHTML = `<div style="white-space:pre-wrap;line-height:1.7;font-size:0.85rem;">${escapeHtml(result.suggestion)}</div>`;
            toast.success('AI 策略已生成');
        } else {
            output.innerHTML = `<p style="color:var(--red);">❌ ${escapeHtml(result.error || '未知错误')}</p>`;
            toast.error('AI 请求失败');
        }
    } catch (e) {
        output.innerHTML = `<p style="color:var(--red);">网络错误: ${escapeHtml(e.message)}</p>`;
        toast.error('网络请求失败');
    } finally {
        btn.disabled = false;
        btn.textContent = '💡 生成捕捉策略';
    }
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('aiBtn').addEventListener('click', generate);
});
