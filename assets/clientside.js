window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.clientside = {
    keydown_listener: function (dummy) {
        document.addEventListener('keydown', function (e) {
            // Do not capture keys if the user is typing in an input or textarea
            const active = document.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
                return;
            }

            // Check if Ctrl+C
            if (e.ctrlKey && (e.key === 'c' || e.key === 'C')) {
                const btn = document.getElementById('ctrl-c-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Ctrl+V
            if (e.ctrlKey && (e.key === 'v' || e.key === 'V')) {
                const btn = document.getElementById('ctrl-v-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Ctrl+Z
            if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
                const btn = document.getElementById('ctrl-z-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Ctrl+Y
            if (e.ctrlKey && (e.key === 'y' || e.key === 'Y')) {
                const btn = document.getElementById('ctrl-y-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Backspace
            if (e.key === 'Backspace') {
                const btn = document.getElementById('backspace-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if N (Add Node)
            if (!e.ctrlKey && !e.metaKey && (e.key === 'n' || e.key === 'N')) {
                const btn = document.getElementById('key-n-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if Shift+E (Remove Edge)
            if (e.shiftKey && (e.key === 'E' || e.key === 'e')) {
                const btn = document.getElementById('key-shift-e-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            // Check if E (Toggle Edge)
            } else if (!e.ctrlKey && !e.metaKey && !e.shiftKey && (e.key === 'e' || e.key === 'E')) {
                const btn = document.getElementById('key-e-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if H (Hadamard)
            if (!e.ctrlKey && !e.metaKey && !e.shiftKey && (e.key === 'h' || e.key === 'H')) {
                const btn = document.getElementById('key-h-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if W (LC)
            if (!e.ctrlKey && !e.metaKey && !e.shiftKey && (e.key === 'w' || e.key === 'W')) {
                const btn = document.getElementById('key-w-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
            // Check if G (Group / Ungroup)
            if (!e.ctrlKey && !e.metaKey && !e.shiftKey && (e.key === 'g' || e.key === 'G')) {
                const btn = document.getElementById('key-g-btn');
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
        }, true);
        return window.dash_clientside.no_update;
    }
};
