/**
 * ChatREL Interactive Q&A
 * Rule-based query answering for relationship reports
 */

// Get report data from embedded JSON
// Check if reportData already exists (loaded by report.js)
let reportData = window.reportData;

if (!reportData) {
    try {
        reportData = JSON.parse(document.getElementById('report-data').textContent);
        window.reportData = reportData; // Make available globally
    } catch (e) {
        console.error('Failed to load report data for Q&A');
    }
}

// Chat state
const chatHistory = [];

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', initializeQueryChat);

function initializeQueryChat() {
    const chatSection = document.getElementById('query-chat-section');
    if (!chatSection || !reportData) return;

    const input = document.getElementById('query-input');
    const sendBtn = document.getElementById('query-send');
    const messagesDiv = document.getElementById('query-messages');

    // Send on button click
    sendBtn.addEventListener('click', () => handleSendQuery(input, messagesDiv));

    // Send on Enter key
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendQuery(input, messagesDiv);
        }
    });

    // Show welcome message
    addAssistantMessage(messagesDiv,
        "Hi! I can answer questions about your relationship analysis. Try asking 'What is engagement?' or 'Who texts more?'",
        null
    );

    // Trigger KPI animations if ui_ux.js is loaded
    if (window.animateKPIs) {
        window.animateKPIs();
    }
}

async function handleSendQuery(input, messagesDiv) {
    const question = input.value.trim();
    if (!question) return;

    // Add user message
    addUserMessage(messagesDiv, question);
    chatHistory.push({ role: 'user', content: question });

    // Clear input
    input.value = '';

    // Show loading
    const loadingId = addLoadingMessage(messagesDiv);

    try {
        // Call API
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                report: reportData  // Send report with request
            })
        });

        const result = await response.json();

        // Remove loading
        document.getElementById(loadingId).remove();

        // Add answer
        addAssistantMessage(messagesDiv, result.answer, result);
        chatHistory.push({ role: 'assistant', content: result.answer, data: result });

    } catch (error) {
        console.error('Query error:', error);
        document.getElementById(loadingId).remove();
        addAssistantMessage(messagesDiv,
            "Sorry, I had trouble processing that. Please try again.",
            null
        );
    }

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addUserMessage(container, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'query-message query-message-user';
    msgDiv.innerHTML = `
        <div class="query-bubble query-bubble-user">
            ${escapeHtml(text)}
        </div>
    `;
    container.appendChild(msgDiv);
}

function addAssistantMessage(container, text, data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'query-message query-message-assistant';

    let html = `
        <div class="query-bubble query-bubble-assistant">
            ${escapeHtml(text)}
        </div>
    `;

    // Add provenance if available
    if (data && data.provenance && data.provenance.length > 0) {
        html += '<div class="query-provenance">';
        html += '<button class="query-provenance-toggle" onclick="this.nextElementSibling.classList.toggle(\'hidden\')">📊 Evidence</button>';
        html += '<div class="query-provenance-list hidden">';
        data.provenance.forEach(item => {
            html += `<div class="query-provenance-item">`;
            html += `<code>${item.key}</code>: ${formatValue(item.value)}`;
            html += `</div>`;
        });
        html += '</div></div>';
    }

    msgDiv.innerHTML = html;
    container.appendChild(msgDiv);
}

function addLoadingMessage(container) {
    const loadingId = 'loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.id = loadingId;
    msgDiv.className = 'query-message query-message-assistant';
    msgDiv.innerHTML = `
        <div class="query-bubble query-bubble-assistant">
            <span class="query-loading">●</span>
            <span class="query-loading" style="animation-delay: 0 .2s;">●</span>
            <span class="query-loading" style="animation-delay: 0.4s;">●</span>
        </div>
    `;
    container.appendChild(msgDiv);
    return loadingId;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatValue(value) {
    if (typeof value === 'number') {
        return value.toFixed(2);
    }
    return String(value);
}
