/* Multi-Tool Agent Frontend */
'use strict';

const API_BASE = '';   // relative — served from same origin

let currentConversationId = null;
let currentProvider = 'anthropic';
let currentModel = null;
let modelsData = null;
let taskHistory = [];  // [{task_id, preview, status, provider, model}]

// ── Bootstrap ──────────────────────────────────────────────────────────────

async function init() {
  await loadModels();
  await checkHealth();
  await loadHistory();
  document.getElementById('provider-select').addEventListener('change', onProviderChange);
}

// ── History loading ────────────────────────────────────────────────────────

async function loadHistory() {
  try {
    const resp = await fetch(`${API_BASE}/tasks?limit=50`);
    if (!resp.ok) return;
    const tasks = await resp.json();
    taskHistory = tasks.map(t => ({
      task_id: t.task_id,
      conversation_id: t.conversation_id,
      preview: (t.input_text || '').slice(0, 60),
      status: t.status,
      provider: t.provider,
      model: t.model,
    }));
    renderHistory();
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

// ── Model loading ──────────────────────────────────────────────────────────

async function loadModels() {
  try {
    const resp = await fetch(`${API_BASE}/models`);
    modelsData = await resp.json();
    onProviderChange();
  } catch (err) {
    console.error('Failed to load models:', err);
    setProviderWarning('Could not reach the API server.');
  }
}

function onProviderChange() {
  const providerEl = document.getElementById('provider-select');
  const modelEl = document.getElementById('model-select');
  const submitBtn = document.getElementById('submit-btn');
  const warningEl = document.getElementById('provider-warning');

  currentProvider = providerEl.value;
  if (!modelsData) return;

  const info = modelsData[currentProvider];
  if (!info) return;

  // Populate model dropdown
  modelEl.innerHTML = '';
  if (info.models.length === 0) {
    modelEl.innerHTML = '<option value="">No models available</option>';
  } else {
    info.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      if (m === info.default_model) opt.selected = true;
      modelEl.appendChild(opt);
    });
  }

  currentModel = modelEl.value || null;
  modelEl.addEventListener('change', () => { currentModel = modelEl.value; });

  // Show warning and disable submit if provider is not configured
  if (!info.configured) {
    warningEl.textContent = currentProvider === 'ollama'
      ? 'Ollama is unreachable. Make sure the Ollama service is running.'
      : `No API key configured for ${currentProvider}. Add it to your .env file.`;
    warningEl.classList.remove('hidden');
    submitBtn.disabled = true;
  } else {
    warningEl.classList.add('hidden');
    submitBtn.disabled = false;
  }
}

function setProviderWarning(msg) {
  const el = document.getElementById('provider-warning');
  el.textContent = msg;
  el.classList.remove('hidden');
}

// ── Health check ───────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`);
    const data = await resp.json();
    const badge = document.getElementById('uptime-badge');
    badge.textContent = `v${data.version} · up ${formatSeconds(data.uptime_seconds)}`;
    badge.className = 'badge ok';
  } catch (_) {
    const badge = document.getElementById('uptime-badge');
    badge.textContent = 'API unreachable';
    badge.className = 'badge error';
  }
}

// ── Submit task ────────────────────────────────────────────────────────────

async function submitTask() {
  const taskInputEl = document.getElementById('task-input');
  const taskInput = taskInputEl.value.trim();
  if (!taskInput) { alert('Please enter a task.'); return; }

  const info = modelsData && modelsData[currentProvider];
  if (info && !info.configured) {
    const msg = currentProvider === 'ollama'
      ? 'Ollama is unreachable. Make sure the Ollama service is running.'
      : `No API key configured for ${currentProvider}. Add it to your .env file.`;
    renderError(msg);
    return;
  }

  const submitBtn = document.getElementById('submit-btn');
  const providerEl = document.getElementById('provider-select');
  const modelEl = document.getElementById('model-select');
  const clearConvBtn = document.getElementById('clear-conv-btn');

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span>Running...';
  providerEl.disabled = true;
  modelEl.disabled = true;
  taskInputEl.disabled = true;
  clearConvBtn.disabled = true;

  showResultLoading(taskInput);

  try {
    const body = {
      task: taskInput,
      provider: currentProvider,
      model: currentModel || undefined,
      conversation_id: currentConversationId || undefined,
    };

    const resp = await fetch(`${API_BASE}/task`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await resp.json();

    if (!resp.ok) {
      renderError(data.detail || 'Unknown error');
      return;
    }

    currentConversationId = data.conversation_id;
    updateConvLabel();
    renderResult(data);
    addToHistory(data, taskInput);

  } catch (err) {
    renderError(String(err));
  } finally {
    const currentInfo = modelsData && modelsData[currentProvider];
    submitBtn.disabled = !!(currentInfo && !currentInfo.configured);
    submitBtn.textContent = 'Submit Task';
    providerEl.disabled = false;
    modelEl.disabled = false;
    taskInputEl.disabled = false;
    clearConvBtn.disabled = false;
  }
}

// ── Conversation ───────────────────────────────────────────────────────────

function clearConversation() {
  if (document.getElementById('clear-conv-btn').disabled) return;
  currentConversationId = null;
  updateConvLabel();
}

function resumeConversation(conversationId) {
  currentConversationId = conversationId;
  updateConvLabel();
  document.getElementById('task-input').focus();
}

function updateConvLabel() {
  const el = document.getElementById('conv-label');
  el.textContent = currentConversationId
    ? `Active conversation: ${currentConversationId.slice(0, 8)}…`
    : 'No active conversation — each task is independent';
}

// ── Render helpers ─────────────────────────────────────────────────────────

function showResultLoading(task) {
  const card = document.getElementById('result-card');
  card.classList.remove('hidden');
  document.getElementById('result-status').className = 'badge loading';
  document.getElementById('result-status').textContent = 'Running…';
  document.getElementById('result-model').textContent = `${currentProvider} / ${currentModel || 'default'}`;
  document.getElementById('result-latency').textContent = '';
  document.getElementById('result-task-id').textContent = '';
  document.getElementById('result-tokens').textContent = '';
  document.getElementById('result-answer').textContent = '';
  document.getElementById('trace-list').innerHTML = '';
  document.getElementById('trace-count').textContent = '0';
}

function renderResult(data) {
  const card = document.getElementById('result-card');
  card.classList.remove('hidden');

  // Status badge
  const statusEl = document.getElementById('result-status');
  statusEl.textContent = data.status.toUpperCase();
  statusEl.className = `badge ${data.status === 'completed' ? 'completed' : 'error'}`;

  // Meta chips
  document.getElementById('result-model').textContent = `${data.provider} / ${data.model}`;
  document.getElementById('result-latency').textContent = `${(data.total_latency_ms / 1000).toFixed(2)}s`;
  document.getElementById('result-task-id').textContent = '';

  // Tokens
  const totalTokens = (data.total_input_tokens || 0) + (data.total_output_tokens || 0);
  document.getElementById('result-tokens').textContent =
    `Tokens: ${data.total_input_tokens} in / ${data.total_output_tokens} out / ${totalTokens} total`;

  // Answer
  const answerEl = document.getElementById('result-answer');
  if (data.status === 'error' && data.error_message) {
    answerEl.textContent = `Error: ${data.error_message}`;
    answerEl.style.color = '#991b1b';
  } else {
    answerEl.textContent = data.final_answer || '(no answer)';
    answerEl.style.color = '';
  }

  // Trace
  renderTrace(data.trace || []);
}

function renderTrace(steps) {
  const listEl = document.getElementById('trace-list');
  const countEl = document.getElementById('trace-count');
  listEl.innerHTML = '';
  countEl.textContent = steps.length;

  if (steps.length === 0) {
    listEl.innerHTML = '<div style="padding:0.75rem 1rem;color:#9ca3af;font-size:0.85rem;">No tool calls were made.</div>';
    return;
  }

  steps.forEach((step, i) => {
    const div = document.createElement('div');
    div.className = 'trace-step';

    const hasError = !!step.tool_error;
    div.innerHTML = `
      <div class="trace-step-header">
        <span class="iteration-label">Iter ${step.iteration}.${step.step_order + 1}</span>
        <span class="tool-badge ${hasError ? 'error' : ''}">${escHtml(step.tool_name)}</span>
        <span class="latency-badge">${step.latency_ms.toFixed(1)}ms</span>
      </div>
      ${step.thinking ? `<div class="trace-thinking">💭 ${escHtml(step.thinking)}</div>` : ''}
      <div class="trace-field">
        <label>Input</label>
        <pre>${escHtml(JSON.stringify(step.tool_input, null, 2))}</pre>
      </div>
      ${hasError
        ? `<div class="trace-field"><label>Error</label><pre style="color:#991b1b">${escHtml(step.tool_error)}</pre></div>`
        : `<div class="trace-field"><label>Output</label><pre>${escHtml(
            step.tool_output !== null && step.tool_output !== undefined
              ? (typeof step.tool_output === 'string' ? step.tool_output : JSON.stringify(step.tool_output, null, 2))
              : '(none)'
          )}</pre></div>`
      }
    `;
    listEl.appendChild(div);
  });
}

function renderError(msg) {
  const card = document.getElementById('result-card');
  card.classList.remove('hidden');
  document.getElementById('result-status').className = 'badge error';
  document.getElementById('result-status').textContent = 'ERROR';
  document.getElementById('result-answer').textContent = msg;
  document.getElementById('result-answer').style.color = '#991b1b';
}

// ── History ────────────────────────────────────────────────────────────────

function addToHistory(data, preview) {
  taskHistory.unshift({
    task_id: data.task_id,
    conversation_id: data.conversation_id,
    preview: preview.slice(0, 60),
    status: data.status,
    provider: data.provider,
    model: data.model,
  });
  renderHistory();
}

function renderHistory() {
  const listEl = document.getElementById('history-list');
  if (taskHistory.length === 0) {
    listEl.innerHTML = '<span class="history-empty">No tasks yet.</span>';
    return;
  }
  listEl.innerHTML = taskHistory.slice(0, 20).map(item => `
    <div class="history-list-item">
      <span class="badge ${item.status === 'completed' ? 'completed' : 'error'}" style="font-size:0.7rem;padding:0.1rem 0.5rem">${item.status}</span>
      <span class="history-task-preview" style="flex:1">${escHtml(item.preview)}</span>
      <span class="history-meta">${escHtml(item.provider)}/${escHtml(item.model)}</span>
      <button class="btn-secondary" style="font-size:0.75rem;padding:0.15rem 0.6rem;margin-left:0.5rem" onclick="loadTask('${escHtml(item.task_id)}')">Retrieve task</button>
      ${item.conversation_id ? `<button class="btn-secondary" style="font-size:0.75rem;padding:0.15rem 0.6rem;margin-left:0.5rem" onclick="resumeConversation('${escHtml(item.conversation_id)}')">Continue conversation</button>` : ''}
    </div>
  `).join('');
}

async function loadTask(taskId) {
  try {
    const resp = await fetch(`${API_BASE}/tasks/${taskId}`);
    if (!resp.ok) { alert('Task not found'); return; }
    const data = await resp.json();
    renderResult(data);
    document.getElementById('result-card').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert('Failed to load task: ' + err);
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────

function escHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatSeconds(s) {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// ── Start ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
