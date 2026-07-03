/**
 * 无人机检测 — API 通信模块
 *
 * 封装所有检测相关的 HTTP API 调用。
 * 支持双目视频源 + 多平台选择。
 */

const BASE = '/api/detection';

export const DetectionAPI = {
    /**
     * 启动双目检测
     * @param {Object} sourceA - 摄像头 A（左目）视频源
     * @param {Object} sourceB - 摄像头 B（右目）视频源
     * @param {Object} config - {platform_id, confidence, frame_interval, ...}
     */
    async start(sourceA, sourceB, config = {}) {
        // 两个都是文件：FormData 上传
        if (sourceA.type === 'file' && sourceB.type === 'file') {
            const fd = new FormData();
            fd.append('video_a', sourceA.value);
            fd.append('video_b', sourceB.value);
            fd.append('config', JSON.stringify(config));
            const r = await fetch(`${BASE}/start`, { method: 'POST', body: fd });
            return r.json();
        }
        // 已有文件路径（恢复的 session）或流地址：JSON 传输
        const body = {
            source_a: sourceA.type === 'file' ? sourceA.value.name : sourceA.value,
            source_b: sourceB.type === 'file' ? sourceB.value.name : sourceB.value,
            ...config,
        };
        const r = await fetch(`${BASE}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return r.json();
    },

    async pause() {
        const r = await fetch(`${BASE}/pause`, { method: 'POST' });
        return r.json();
    },

    async resume() {
        const r = await fetch(`${BASE}/resume`, { method: 'POST' });
        return r.json();
    },

    async stop() {
        const r = await fetch(`${BASE}/stop`, { method: 'POST' });
        return r.json();
    },

    async status(sessionId) {
        const q = sessionId ? `?session_id=${sessionId}` : '';
        const r = await fetch(`${BASE}/status${q}`);
        return r.json();
    },

    async tracks(format = 'summary') {
        const r = await fetch(`${BASE}/tracks?format=${format}`);
        return r.json();
    },

    /** 获取指定摄像头的预览帧 URL */
    previewUrl(channel, sessionId) {
        const q = sessionId ? `&session_id=${sessionId}` : '';
        return `${BASE}/preview?channel=${channel}${q}`;
    },

    /** 列出 data/fact/ 中的检测文件 */
    async listFiles() {
        const r = await fetch(`${BASE}/files`);
        return r.json();
    },

    /** 删除指定检测文件 */
    async deleteFile(filename) {
        const r = await fetch(`${BASE}/files/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        return r.json();
    },

    async saveConfig(config) {
        const r = await fetch(`${BASE}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        return r.json();
    },

    async getConfig() {
        const r = await fetch(`${BASE}/config`);
        return r.json();
    },

    /**
     * 保存检测结果到指定平台
     * @param {string} [sessionId] - 会话 ID
     * @param {string} [platformId] - 目标平台 ID，默认使用会话中记录的
     */
    async saveResults(sessionId, platformId) {
        const body = {};
        if (sessionId) body.session_id = sessionId;
        if (platformId) body.platform_id = platformId;
        const r = await fetch(`${BASE}/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return r.json();
    },
};
