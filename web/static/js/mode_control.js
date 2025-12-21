document.addEventListener('DOMContentLoaded', function () {
    const modeBtn = document.getElementById('modeDropdownBtn');
    const modeMenu = document.getElementById('modeDropdownMenu');
    const modeOptions = document.querySelectorAll('.mode-option');
    const currentModeSpan = document.getElementById('currentModeLabel');
    const modeInfo = document.getElementById('modeInfoText');
    const uploadForm = document.getElementById('uploadForm');

    // Create hidden input for mode
    const modeInput = document.createElement('input');
    modeInput.type = 'hidden';
    modeInput.name = 'mode';
    modeInput.id = 'modeInput';
    if (uploadForm) {
        uploadForm.appendChild(modeInput);
    }

    // Default mode from server or local storage
    let currentMode = localStorage.getItem('chatrel_mode') || 'normal';

    // Mode descriptions
    const descriptions = {
        'normal': 'Standard analysis (config CSM determines behavior)',
        'csm': 'Enable Contextual Sentiment Memory (learning) for this run',
        'demo': 'Demo mode — results cached per-chat for instant replay. No learning updates.'
    };

    // Initialize
    setMode(currentMode, false);

    // Toggle dropdown
    modeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        modeMenu.classList.toggle('show');
    });

    // Close dropdown on click outside
    document.addEventListener('click', () => {
        modeMenu.classList.remove('show');
    });

    modeMenu.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // Handle selection
    modeOptions.forEach(option => {
        option.addEventListener('click', () => {
            const mode = option.dataset.mode;
            if (mode === 'demo' && currentMode !== 'demo') {
                if (confirm('Demo Mode: any chat you upload while demo mode is on will be stored and reused. Do you want to proceed?')) {
                    setMode(mode);
                }
            } else {
                setMode(mode);
            }
            modeMenu.classList.remove('show');
        });

        // Tooltip on hover
        option.addEventListener('mouseenter', () => {
            modeInfo.textContent = descriptions[option.dataset.mode];
        });
    });

    function setMode(mode, save = true) {
        currentMode = mode;
        if (save) localStorage.setItem('chatrel_mode', mode);

        // Update UI
        modeOptions.forEach(opt => {
            opt.classList.toggle('active', opt.dataset.mode === mode);
        });

        const labels = {
            'normal': 'Normal',
            'csm': 'CSM Mode',
            'demo': 'Demo Mode'
        };

        currentModeSpan.textContent = labels[mode];
        modeInput.value = mode;

        // Update body class for styling
        if (mode === 'demo') {
            document.body.classList.add('demo-mode-active');
            showToast('Demo Mode Active');
        } else {
            document.body.classList.remove('demo-mode-active');
        }
    }

    function showToast(message) {
        // Remove existing toast
        const existing = document.querySelector('.demo-toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = 'demo-toast show';
        toast.innerHTML = `<span>⚡</span> <span>${message}</span>`;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
