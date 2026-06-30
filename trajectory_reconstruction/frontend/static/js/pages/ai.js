/**
 * AI Page — 捕捉策略生成
 * - sessionStorage 持久化：切换页面不丢失文本
 * - 滚动容器：长文本自动滚动
 * - 保存报告：写入 reports/ 目录
 */
import { toast } from '../common/toast.js';

const { methodsData } = window._PAGE_DATA_ || {};
const STORAGE_KEY = 'ai_suggestion_text';

// ═══════ 恢复历史文本 ═══════
const outputEl = document.getElementById('aiOutput');
const saveBar = document.getElementById('saveBar');

const savedText = sessionStorage.getItem(STORAGE_KEY);
if (savedText) {
    outputEl.textContent = savedText;
    saveBar.style.display = 'flex';
}

// ═══════ 生成 ═══════
async function generate() {
    const btn = document.getElementById('aiBtn');
    btn.disabled = true; btn.textContent = 'AI 分析中...';
    outputEl.textContent = '';
    saveBar.style.display = 'none';

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
        btn.disabled = false; btn.textContent = '生成捕捉策略';
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
            outputEl.textContent = result.suggestion;
            sessionStorage.setItem(STORAGE_KEY, result.suggestion);
            saveBar.style.display = 'flex';
            toast.success('AI 策略已生成');
        } else {
            outputEl.textContent = '请求失败: ' + (result.error || '未知错误');
            sessionStorage.removeItem(STORAGE_KEY);
            toast.error('AI 请求失败');
        }
    } catch (e) {
        outputEl.textContent = '网络错误: ' + e.message;
        sessionStorage.removeItem(STORAGE_KEY);
        toast.error('网络请求失败');
    } finally {
        btn.disabled = false; btn.textContent = '生成捕捉策略';
    }
}

// ═══════ 保存报告 ═══════
async function saveReport() {
    const content = outputEl.textContent.trim();
    if (!content) return toast.warning('无内容可保存');

    const checks = document.querySelectorAll('.ai-platform-check:checked');
    const names = [];
    checks.forEach(cb => {
        if (methodsData[cb.value]) names.push(methodsData[cb.value].name);
    });

    const btn = document.getElementById('saveReportBtn');
    btn.disabled = true; btn.textContent = '保存中...';

    try {
        const resp = await fetch('/api/save_report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, platforms: names.join(', ') }),
        });
        const result = await resp.json();
        if (result.success) {
            document.getElementById('savedPath').textContent = '已保存: ' + result.filepath;
            toast.success('报告已保存');
        } else {
            toast.error('保存失败: ' + result.error);
        }
    } catch (e) {
        toast.error('网络错误: ' + e.message);
    } finally {
        btn.disabled = false; btn.innerHTML = '<svg width="1em" height="1em" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:text-bottom;"><path d="M8 2.5V11" fill="none"/><polyline points="5,8.5 8,11.5 11,8.5" fill="none"/><path d="M2.5 13.5h11" fill="none"/></svg> 保存报告';
    }
}

// ═══════ 初始化 ═══════
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('aiBtn').addEventListener('click', generate);
    document.getElementById('saveReportBtn').addEventListener('click', saveReport);
});
