/** 轻量 toast 通知 */
let _container = null;
function ensure() {
    if (!_container) {
        _container = document.getElementById('toastContainer');
        if (!_container) {
            _container = document.createElement('div');
            _container.className = 'toast-container';
            document.body.appendChild(_container);
        }
    }
    return _container;
}
export function show(msg, type = '', dur = 2500) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    ensure().appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 400); }, dur);
}
export const toast = { show, success: m => show(m,'success'), error: m => show(m,'error'), warning: m => show(m,'warning') };
