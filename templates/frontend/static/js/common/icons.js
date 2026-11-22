// 简洁风格图标（自绘 SVG，1em 尺寸，跟随文字颜色）
export const ICONS = {
    check: '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-0.14em"><path d="M3.2 8.6l3.1 3.1 6.5-7.2"/></svg>',
    dot: '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="currentColor" style="vertical-align:-0.12em"><circle cx="8" cy="8" r="3.2"/></svg>',
    play: '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="currentColor" style="vertical-align:-0.14em"><path d="M5 3.6l7.2 4.4L5 12.4z"/></svg>',
    flag: '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-0.14em"><path d="M4 14V2.5"/><path d="M4 3h7.5l-1.6 2.6L11.5 8H4"/></svg>',
    crosshair: '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" style="vertical-align:-0.14em"><circle cx="8" cy="8" r="4.4"/><path d="M8 1.5v2.4M8 12.1v2.4M1.5 8h2.4M12.1 8h2.4"/></svg>',
};

export function icon(name) {
    return ICONS[name] || '';
}
