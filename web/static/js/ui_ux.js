/**
 * ChatREL UI/UX Enhancements
 * Mode dropdown, demo badge, KPI animations, toasts, and chart transitions
 */

(function () {
    'use strict';

    // === CONSTANTS ===
    const STORAGE_KEY_MODE = 'chatrel_mode';
    const TOAST_DURATION = 4000; // 4 seconds
    const KPI_ANIMATION_DURATION = 1200; // 1.2 seconds

    // === STATE ===
    let currentMode = 'normal';
    let toastContainer = null;

    // === INITIALIZATION ===
    document.addEventListener('DOMContentLoaded', init);

    function init() {
        initToastContainer();
        initModeDropdown();
        initKPIAnimations();
        initChartFadeIns();
    }

    // === TOAST SYSTEM ===
    function initToastContainer() {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<div class="toast-content">${escapeHtml(message)}</div>`;

        toastContainer.appendChild(toast);

        // Auto-dismiss after duration
        setTimeout(() => {
            toast.classList.add('toast-exit');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 250); // Wait for exit animation
        }, TOAST_DURATION);
    }

    // === MODE DROPDOWN ===
    function initModeDropdown() {
        const dropdown = document.getElementById('mode-dropdown');
        if (!dropdown) return;

        // Load saved mode from localStorage
        const savedMode = localStorage.getItem(STORAGE_KEY_MODE) || 'normal';
        currentMode = savedMode;
        dropdown.value = savedMode;

        // Update demo badge visibility
        updateDemoBadgeVisibility();

        // Listen for changes
        dropdown.addEventListener('change', handleModeChange);

        // Inject hidden input into upload forms (for mode submission)
        injectModeInputIntoForms();
    }

    function handleModeChange(event) {
        const newMode = event.target.value;
        const previousMode = currentMode;

        if (newMode === 'demo' && previousMode !== 'demo') {
            // Show confirmation modal for Demo Mode
            showDemoConfirmation(() => {
                // Confirmed
                setMode(newMode);
                showToast('🎉 Demo Mode activated! Upload a chat to cache results for instant replay.', 'info');
            }, () => {
                // Canceled - revert dropdown
                event.target.value = previousMode;
            });
        } else {
            setMode(newMode);
        }
    }

    function setMode(mode) {
        currentMode = mode;
        localStorage.setItem(STORAGE_KEY_MODE, mode);
        updateDemoBadgeVisibility();
        updateFormModeInputs();
    }

    function updateDemoBadgeVisibility() {
        const badge = document.getElementById('demo-badge');
        if (!badge) return;

        if (currentMode === 'demo') {
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    }

    function injectModeInputIntoForms() {
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            let input = form.querySelector('input[name="mode"]');
            if (!input) {
                input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'mode';
                form.appendChild(input);
            }
            input.value = currentMode;
        });
    }

    function updateFormModeInputs() {
        const modeInputs = document.querySelectorAll('input[name="mode"]');
        modeInputs.forEach(input => {
            input.value = currentMode;
        });
    }

    // === DEMO CONFIRMATION MODAL ===
    function showDemoConfirmation(onConfirm, onCancel) {
        // Create modal backdrop
        const backdrop = document.createElement('div');
        backdrop.className = 'modal-backdrop';
        backdrop.innerHTML = `
            <div class="modal">
                <h2 class="modal-title">Enable Demo Mode?</h2>
                <div class="modal-body">
                    Demo Mode caches analysis results for instant replay. Great for presentations and demos!
                    <br><br>
                    <strong>How it works:</strong> Your first upload will run normally and cache the result. 
                    Subsequent uploads of the same file will replay instantly from cache.
                </div>
                <div class="modal-actions">
                    <button class="modal-btn modal-btn-secondary" data-action="cancel">Cancel</button>
                    <button class="modal-btn modal-btn-primary" data-action="confirm">Enable Demo Mode</button>
                </div>
            </div>
        `;

        document.body.appendChild(backdrop);

        // Handle button clicks
        backdrop.querySelector('[data-action="confirm"]').addEventListener('click', () => {
            document.body.removeChild(backdrop);
            onConfirm();
        });

        backdrop.querySelector('[data-action="cancel"]').addEventListener('click', () => {
            document.body.removeChild(backdrop);
            onCancel();
        });

        // Handle Escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                document.body.removeChild(backdrop);
                onCancel();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);

        // Handle backdrop click
        backdrop.addEventListener('click', (e) => {
            if (e.target === backdrop) {
                document.body.removeChild(backdrop);
                onCancel();
            }
        });
    }

    // === KPI COUNT ANIMATIONS ===
    function initKPIAnimations() {
        const kpiElements = document.querySelectorAll('[data-animate-count]');

        kpiElements.forEach(el => {
            const targetValue = parseFloat(el.getAttribute('data-animate-count'));
            if (isNaN(targetValue)) return;

            animateCount(el, 0, targetValue, KPI_ANIMATION_DURATION);
        });
    }

    function animateCount(element, start, end, duration) {
        const startTime = performance.now();
        const isInteger = Number.isInteger(end);

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-out cubic function
            const easeOut = 1 - Math.pow(1 - progress, 3);

            const currentValue = start + (end - start) * easeOut;

            if (isInteger) {
                element.textContent = Math.round(currentValue);
            } else {
                element.textContent = currentValue.toFixed(1);
            }

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                // Ensure final value is exact
                element.textContent = isInteger ? end : end.toFixed(1);
            }
        }

        requestAnimationFrame(update);
    }

    // Export for external access
    window.animateKPIs = initKPIAnimations;

    // === CHART FADE-IN ANIMATIONS ===
    function initChartFadeIns() {
        // Wait for charts to be rendered by report.js
        // We'll use a MutationObserver to detect when canvas context is used

        const chartContainers = document.querySelectorAll('[data-chart-ready="false"]');

        chartContainers.forEach(container => {
            const canvas = container.querySelector('canvas');
            if (!canvas) return;

            // Simple timeout-based approach: wait for Chart.js to render
            setTimeout(() => {
                container.setAttribute('data-chart-ready', 'true');
            }, 300); // Small delay after page load
        });

        // Listen for custom chart rendered events (if report.js dispatches them)
        document.addEventListener('chartRendered', (event) => {
            const canvasId = event.detail?.canvasId;
            if (canvasId) {
                const container = document.getElementById(canvasId + '-container');
                if (container) {
                    container.setAttribute('data-chart-ready', 'true');
                }
            }
        });
    }

    // === DEMO RESULT SAVED NOTIFICATION ===
    // This would be triggered by backend, but we can listen for page load patterns
    // Check if URL has a demo=saved parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('demo') === 'saved') {
        setTimeout(() => {
            showToast('✨ Demo result saved! Next upload of this chat will replay instantly.', 'success');
        }, 500);
    }

    // === UTILITY FUNCTIONS ===
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

})();
