/**
 * Analyst + Assessment Page — 运动学分析 / 捕捉时机 / 智能风险评估
 *
 * 数据来源:
 *   /analysis/data     各平台（含综合轨迹与预测拼接）的运动学指标
 *   /analysis/capture  综合轨迹的最佳捕捉时机
 *   /api/sensors/assess 重建轨迹 + KinFlow 预测 + 六因子风险评估
 */
(function () {
    'use strict';
    const PAGE = window._PAGE_DATA_ || {};
    const COLLAB = PAGE.collab || {};
    const ZONE = PAGE.zone || {};
    const $ = id => document.getElementById(id);

    const STATUS_COLOR = { normal: '#52c41a', degraded: '#f6c445', lost: '#ff4d4f' };
    const STATUS_TEXT = { normal: '正常', degraded: '降级', lost: '失联' };
    const PRIORITY_TEXT = { 1: '立即', 2: '优先', 3: '建议', 4: '常规' };
    const PRIORITY_COLOR = { 1: '#ff4d4f', 2: '#ff9f43', 3: '#f6c445', 4: '#52c41a' };

    const charts = {};
    let allData = {};

    function chart(id) {
        if (!charts[id]) charts[id] = echarts.init($(id));
        return charts[id];
    }

    function fmt(v, d, unit) {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
        return Number(v).toFixed(d) + (unit || '');
    }

    function row(label, value, color) {
        return '<div style="display:flex;align-items:baseline;gap:8px;padding:2px 0;">'
            + '<span style="flex:0 0 auto;white-space:nowrap;">' + label + '</span>'
            + '<b style="flex:1 1 auto;text-align:right;word-break:break-word;'
            + 'color:' + (color || 'var(--text-primary)') + ';font-family:var(--mono);font-weight:600;">'
            + value + '</b></div>';
    }

    // ═══════════════ 运动学分析 ═══════════════

    const LEGEND_ORDER = ['visible', 'infrared', 'radar', 'self', 'synthetic'];

    async function loadAnalysis() {
        try {
            allData = await (await fetch('/analysis/data')).json();
        } catch (e) {
            allData = {};
        }
        renderLegend();
        renderCharts();
    }

    function renderLegend() {
        const el = $('legend');
        const items = [];
        for (const mid of LEGEND_ORDER) if (allData[mid]) items.push([mid, allData[mid]]);
        for (const [mid, info] of Object.entries(allData)) {
            if (!LEGEND_ORDER.includes(mid)) items.push([mid, info]);
        }
        el.innerHTML = items.map(([id, info]) =>
            '<div class="legend-item"><span class="legend-dot" style="background:' + info.color
            + ';color:' + info.color + ';"></span>' + info.name + '</div>').join('');
    }

    function baseAxis(name, unit) {
        return {
            tooltip: {
                trigger: 'axis', backgroundColor: '#21262d', borderColor: '#30363d',
                textStyle: { color: '#e6edf3', fontSize: 12 },
                valueFormatter: v => (v != null ? Number(v).toFixed(2) : '-'),
            },
            grid: { top: 12, bottom: 28, left: 8, right: 18, containLabel: true },
            xAxis: {
                type: 'value', axisLabel: { color: '#8b949e', fontSize: 9 },
                axisLine: { lineStyle: { color: '#30363d' } }, splitLine: { show: false },
            },
            yAxis: {
                type: 'value', name: name + (unit ? ' (' + unit + ')' : ''),
                nameTextStyle: { color: '#8b949e', fontSize: 10 },
                axisLabel: { color: '#8b949e', fontSize: 10 },
                splitLine: { lineStyle: { color: '#21262d' } },
            },
        };
    }

    function seriesFor(field) {
        return Object.entries(allData).map(([id, info]) => ({
            name: info.name,
            type: 'line',
            symbol: 'none',
            smooth: true,
            data: (info.time_steps || [])
                .map((t, i) => [t, (info[field] || [])[i]])
                .filter(p => p[1] != null),
            lineStyle: { color: info.color, width: 2 },
            itemStyle: { color: info.color },
        }));
    }

    function renderCharts() {
        const defs = [
            ['heightChart', '高度', 'm', 'heights'],
            ['speedChart', '速度', 'm/s', 'speeds'],
            ['accelChart', '加速度', 'm/s²', 'accelerations'],
            ['curvatureChart', '曲率', '1/m', 'curvatures'],
        ];
        for (const [id, name, unit, field] of defs) {
            const opt = baseAxis(name, unit);
            opt.series = seriesFor(field);
            chart(id).setOption(opt, true);
        }
    }

    async function loadCapture() {
        const el = $('captureResult');
        try {
            const r = await (await fetch('/analysis/capture')).json();
            if (!Array.isArray(r) || !r.length) {
                el.innerHTML = '<span style="color:var(--text-tertiary);">请先在预测页预测综合轨迹</span>';
                return;
            }
            const medals = ['#f6c445', '#b1bac4', '#cd7f32'];
            el.innerHTML = '<div class="capture-grid">' + r.map((v, i) =>
                '<div class="capture-item">'
                + '<div class="capture-rank" style="color:' + (medals[i] || 'var(--text-tertiary)') + ';">'
                + '#' + (i + 1) + ' 候选捕捉点</div>'
                + '<div class="capture-pos">(' + v.position.join(', ') + ')</div>'
                + '<div class="capture-meta">'
                + '<span>时间 ' + v.time + ' s</span>'
                + '<span>高度 ' + v.height + ' m</span>'
                + '<span>速度 ' + v.speed + ' m/s</span>'
                + '<span style="color:var(--green);font-weight:600;">得分 ' + v.score + '</span>'
                + '</div></div>').join('') + '</div>';
        } catch (e) {
            el.innerHTML = '<span style="color:var(--red);">加载失败</span>';
        }
    }

    // ═══════════════ 智能评估 ═══════════════

    function initRiskInputs() {
        const c = ZONE.center || [8, 2, 3];
        $('zx').value = c[0];
        $('zy').value = c[1];
        $('zz').value = c[2];
        $('radius').value = ZONE.radius != null ? ZONE.radius : 4;
        $('ymax').value = ZONE.y_max != null ? ZONE.y_max : 20;
        $('horizon').value = COLLAB.prediction_horizon_s || 2.0;
    }

    function zonePayload() {
        return {
            center: [Number($('zx').value), Number($('zy').value), Number($('zz').value)],
            radius: Number($('radius').value),
            y_min: 0,
            y_max: Number($('ymax').value),
        };
    }

    function renderGauge(score, color) {
        chart('riskGauge').setOption({
            animationDuration: 600,
            series: [{
                type: 'gauge', min: 0, max: 100, radius: '92%',
                startAngle: 210, endAngle: -30,
                axisLine: {
                    lineStyle: {
                        width: 13,
                        color: [[0.25, '#52c41a'], [0.5, '#f6c445'], [0.75, '#ff9f43'], [1, '#ff4d4f']],
                    },
                },
                pointer: { width: 4, length: '62%', itemStyle: { color: 'auto' } },
                axisTick: { show: false }, splitLine: { show: false }, axisLabel: { show: false },
                anchor: { show: true, size: 8, itemStyle: { color: '#8b949e' } },
                title: { show: false },
                detail: {
                    valueAnimation: true, fontSize: 28, offsetCenter: [0, '42%'],
                    color: color || '#e6edf3', formatter: v => v.toFixed(0),
                },
                data: [{ value: score }],
            }],
        }, true);
    }

    function renderFactors(factors) {
        $('factorList').innerHTML = factors.map(f => {
            const pct = Math.min(f.score * 100, 100);
            const color = pct >= 75 ? '#ff4d4f' : pct >= 50 ? '#ff9f43' : pct >= 25 ? '#f6c445' : '#52c41a';
            return '<div style="margin-bottom:10px;">'
                + '<div style="display:flex;justify-content:space-between;gap:8px;">'
                + '<span><b style="color:var(--text-primary);">' + f.label + '</b>'
                + '<span style="color:var(--text-tertiary);font-size:0.72rem;"> 权重 ' + f.weight.toFixed(2) + '</span></span>'
                + '<span style="font-family:var(--mono);">' + f.score.toFixed(2) + ' · 贡献 ' + f.contribution.toFixed(1) + '</span></div>'
                + '<div style="height:6px;border-radius:3px;background:var(--border-muted);margin:3px 0 2px;overflow:hidden;">'
                + '<div style="height:100%;width:' + pct + '%;background:' + color + ';border-radius:3px;"></div></div>'
                + '<div style="font-size:0.72rem;color:var(--text-tertiary);">' + f.detail + '</div></div>';
        }).join('');
    }

    function renderRecommendations(recs) {
        $('recommendList').innerHTML = recs.map((r, i) => {
            const color = PRIORITY_COLOR[r.priority] || '#888';
            const border = i < recs.length - 1 ? 'border-bottom:1px solid var(--border-muted);' : '';
            return '<div style="display:flex;gap:8px;padding:7px 0;' + border + '">'
                + '<span class="badge" style="background:' + color + '22;color:' + color + ';flex:0 0 auto;">'
                + (PRIORITY_TEXT[r.priority] || '') + '</span>'
                + '<div><div style="color:var(--text-primary);font-weight:600;">' + r.action + '</div>'
                + '<div style="font-size:0.72rem;color:var(--text-tertiary);">' + r.reason + '</div></div></div>';
        }).join('') || '暂无建议';
    }

    function renderTimeline(timeline, events) {
        const data = (timeline.times || []).map((t, i) => [t, timeline.scores[i]]);
        const markLines = [];
        const tStart = (timeline.times && timeline.times.length) ? timeline.times[0] : 0;
        if (events && events.zone_entry && events.zone_entry.time != null
            && events.zone_entry.time > tStart + 0.3) {
            markLines.push({
                xAxis: events.zone_entry.time,
                label: { formatter: '进入防护区', position: 'insideEndTop', color: '#ff8a8a', fontSize: 10 },
                lineStyle: { color: '#ff6b6b', type: 'dashed' },
            });
        }
        chart('riskTimeline').setOption({
            tooltip: {
                trigger: 'axis', backgroundColor: '#21262d', borderColor: '#30363d',
                textStyle: { color: '#e6edf3', fontSize: 12 },
            },
            grid: { top: 26, bottom: 30, left: 42, right: 18 },
            xAxis: {
                type: 'value', name: '时间 (s)', nameTextStyle: { color: '#8b949e', fontSize: 10 },
                axisLabel: { color: '#8b949e', fontSize: 10 },
                axisLine: { lineStyle: { color: '#30363d' } }, splitLine: { show: false },
            },
            yAxis: {
                type: 'value', min: 0, max: 100, name: '风险分',
                nameTextStyle: { color: '#8b949e', fontSize: 10 },
                axisLabel: { color: '#8b949e', fontSize: 10 },
                splitLine: { lineStyle: { color: '#21262d' } },
            },
            series: [{
                type: 'line', smooth: true, symbol: 'none', data,
                lineStyle: { width: 2, color: '#ff9f43' },
                areaStyle: {
                    opacity: 0.18,
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#ff4d4f' },
                        { offset: 1, color: 'rgba(255,77,79,0.02)' },
                    ]),
                },
                markLine: { symbol: 'none', silent: true, data: markLines },
            }],
        }, true);
    }

    function renderEvidence(result) {
        const ev = result.evidence || {};
        const health = ev.sensor_health || {};
        const conf = ev.confidence != null ? ev.confidence : 0;
        const confColor = conf >= 0.75 ? '#52c41a' : conf >= 0.5 ? '#f6c445' : '#ff4d4f';
        let html = '<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            + '<span>证据置信度</span><b style="color:' + confColor + ';font-family:var(--mono);">'
            + (conf * 100).toFixed(0) + '%</b></div>'
            + '<div style="height:6px;background:var(--border-muted);border-radius:3px;overflow:hidden;margin-bottom:8px;">'
            + '<div style="height:100%;width:' + (conf * 100) + '%;background:' + confColor + ';"></div></div>';
        html += row('平均不确定度', fmt(ev.mean_uncertainty, 2, ' m'))
            + row('融合帧数', (ev.n_frames || 0) + ' 帧 @ ' + fmt(ev.hz, 1, ' Hz'));
        const entries = Object.entries(health);
        if (entries.length) {
            html += '<div style="margin-top:8px;font-weight:600;color:var(--text-primary);">传感器状态</div>';
            html += entries.map(([sid, h]) => {
                const color = STATUS_COLOR[h.status] || '#888';
                return '<div style="display:flex;align-items:center;gap:6px;padding:4px 0;">'
                    + '<span class="legend-dot" style="background:' + (h.color || '#999') + ';color:' + (h.color || '#999') + ';"></span>'
                    + '<b style="color:var(--text-primary);">' + (h.name || sid) + '</b>'
                    + '<span style="color:var(--text-tertiary);">覆盖 ' + (h.coverage * 100).toFixed(0)
                    + '% · 创新量 ' + fmt(h.mean_innovation, 2, 'm') + '</span>'
                    + '<span class="badge" style="margin-left:auto;background:' + color + '22;color:' + color + ';">'
                    + (STATUS_TEXT[h.status] || h.status) + '</span></div>';
            }).join('');
        }
        $('evidenceBox').innerHTML = html;
    }

    function renderEvents(result) {
        const ev = result.events || {};
        const pred = result.prediction || {};
        const pipe = result.pipeline || {};
        const acc = pipe.accuracy;
        let html = '';
        html += row('预测来源', pred.source === 'kinflow' ? 'KinFlow（学习式模型）'
            : pred.source === 'linear' ? '匀速外推（模型不可用）' : '-');
        html += row('预测点数', (pred.positions || []).length + ' 点');
        html += row('进入防护区', ev.zone_entry
            ? fmt(ev.zone_entry.time, 2, ' s') + ' @ [' + (ev.zone_entry.position || []).join(', ') + ']'
            : '未进入');
        html += row('预测入侵时刻', ev.predicted_entry != null ? fmt(ev.predicted_entry, 2, ' s') : '视界内无入侵');
        html += row('风险峰值', ev.peaked && ev.peaked.score != null
            ? fmt(ev.peaked.time, 2, ' s') + ' / ' + fmt(ev.peaked.score, 0) + ' 分' : '-');
        html += row('区内帧数', (ev.n_inside_frames || 0) + ' / ' + ((result.timeline.times || []).length || 0));
        if (pipe.reconstruction) {
            html += row('重建补洞/剔除', (pipe.reconstruction.n_filled || 0) + ' / '
                + (pipe.reconstruction.n_outliers || 0) + ' 帧');
        }
        if (acc && acc.fused_rmse != null) {
            html += '<div style="margin-top:8px;font-weight:600;color:var(--text-primary);">仿真精度（相对真值）</div>';
            html += row('最优单传感器', fmt(acc.best_single_rmse, 3, ' m'));
            html += row('融合 / 重建', fmt(acc.fused_rmse, 3, ' m') + ' / ' + fmt(acc.reconstructed_rmse, 3, ' m'),
                '#4fc3f7');
            if (acc.fusion_gain_pct != null) html += row('协同增益', '+' + fmt(acc.fusion_gain_pct, 1, '%'), '#52c41a');
        }
        $('eventBox').innerHTML = html;
    }

    async function runAssessment() {
        const payload = {
            scenario: $('scenario').value,
            zone: zonePayload(),
            prediction_horizon_s: Number($('horizon').value),
        };
        $('assessBtn').disabled = true;
        $('assessStatus').textContent = '评估中…';
        const t0 = performance.now();
        try {
            const resp = await fetch('/api/sensors/assess', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await resp.json();
            if (!result.success) {
                $('assessStatus').textContent = result.error || '评估失败';
                return;
            }
            $('riskLevel').textContent = result.level || '—';
            $('riskLevel').style.color = result.color || 'var(--text-tertiary)';
            $('riskAdvice').textContent = result.level_advice || '';
            $('riskSummary').textContent = result.summary || '';
            renderGauge(result.score || 0, result.color);
            renderFactors(result.factors || []);
            renderRecommendations(result.recommendations || []);
            renderTimeline(result.timeline || {}, result.events || {});
            renderEvidence(result);
            renderEvents(result);
            $('assessStatus').textContent = '完成：' + (result.scenario_label || '')
                + ' · ' + (performance.now() - t0).toFixed(0) + ' ms';
        } catch (err) {
            $('assessStatus').textContent = '请求失败: ' + err.message;
        } finally {
            $('assessBtn').disabled = false;
        }
    }

    // ═══════════════ 初始化 ═══════════════

    window.addEventListener('resize', () => {
        Object.values(charts).forEach(c => c.resize());
    });

    initRiskInputs();

    // 数据源选择本地记忆，刷新后保持
    const savedScenario = localStorage.getItem('analysis.scenario');
    if (savedScenario && $('scenario')) $('scenario').value = savedScenario;

    $('assessBtn').addEventListener('click', runAssessment);
    $('scenario').addEventListener('change', () => {
        localStorage.setItem('analysis.scenario', $('scenario').value);
        runAssessment();
    });

    loadAnalysis();
    loadCapture();
    runAssessment();

    // 从预测页返回（bfcache）时刷新一次，保证刚生成的预测能用于捕捉时机
    window.addEventListener('pageshow', e => { if (e.persisted) { loadAnalysis(); loadCapture(); } });
})();
