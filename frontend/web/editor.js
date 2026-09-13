/* Voice editing: select text, speak an instruction, an LLM rewrites it.
   A browser stand-in for the eventual desktop hotkeys, so the pipeline and the
   prompt can be iterated on before any of it touches /dev/uinput. */
'use strict';

const $ = (id) => document.getElementById(id);
const LS = { key: 'vv.openrouter.key', model: 'vv.openrouter.model', prompt: 'vv.system.prompt', review: 'vv.review' };

const SAMPLE = `We deployed the new service to cooper netties last Tuesday and it's been running fine since. The main thing that changed is we moved the ingest path off of the old queue and onto red is streams, which cut the tail latency by about a third.

There are still two open questions. First, whether we keep the fallback path around at all. Second, how we want to handle back pressure when the consumer falls behind.`;

const state = {
  busy: false, mode: null, ws: null, audioCtx: null, worklet: null, stream: null,
  heard: '', models: [], serverKey: false, defaultPrompt: '',
  selection: null, undoStack: [], asrMs: null,
  awaitingReview: false, editAbort: null, lastInstruction: null, lastSelection: null,
};

/* True whenever Esc has something to cancel. */
function inFlight() { return state.busy || state.awaitingReview || state.editAbort !== null; }

/* ------------------------------------------------------------------ utils */

function toast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.className = 'toast' + (isError ? ' err' : '');
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, isError ? 7000 : 2800);
}

function setStep(name, cls) {
  const li = document.querySelector(`.steps li[data-step="${name}"]`);
  if (li) li.className = cls || '';
}
function resetSteps() { ['record','asr','llm','apply'].forEach(s => setStep(s, '')); }

function escapeHtml(t) {
  return t.replace(/[&<>"]/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]));
}

/* Word-level LCS diff, enough to show what the model actually touched. */
function diffWords(before, after) {
  const a = before.split(/(\s+)/), b = after.split(/(\s+)/);
  const n = a.length, m = b.length;
  // Cap the table so a huge paste cannot lock the tab up.
  if (n * m > 4_000_000) {
    return `<del>${escapeHtml(before)}</del> <ins>${escapeHtml(after)}</ins>`;
  }
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);

  let out = '', i = 0, j = 0;
  const flush = (tag, buf) => buf ? `<${tag}>${escapeHtml(buf)}</${tag}>` : '';
  let delBuf = '', insBuf = '';
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out += flush('del', delBuf) + flush('ins', insBuf); delBuf = insBuf = '';
      out += escapeHtml(a[i]); i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) { delBuf += a[i++]; }
    else { insBuf += b[j++]; }
  }
  while (i < n) delBuf += a[i++];
  while (j < m) insBuf += b[j++];
  return out + flush('del', delBuf) + flush('ins', insBuf);
}

/* ----------------------------------------------------- instruction panel */

function showInstruction(label, { editable = false, hint = '<kbd>Esc</kbd> cancels' } = {}) {
  $('instruction').hidden = false;
  $('instruction-label').textContent = label;
  $('instruction-hint').innerHTML = hint;
  const box = $('instruction-text');
  box.readOnly = !editable;
  box.classList.toggle('editable', editable);
  $('instruction-actions').hidden = !editable;
  if (editable) { box.focus(); box.setSelectionRange(box.value.length, box.value.length); }
}

function hideInstruction() {
  $('instruction').hidden = true;
  $('instruction-text').value = '';
  $('instruction-actions').hidden = true;
  state.awaitingReview = false;
}

/* One key, one meaning: stop whatever is happening and change nothing. */
function cancelAll(reason) {
  const wasDoing = inFlight();
  if (state.editAbort) { state.editAbort.abort(); state.editAbort = null; }
  stopRecording(false);
  hideInstruction();
  resetControls();
  resetSteps();
  if (wasDoing) toast(reason || 'Cancelled — nothing changed.');
}

/* -------------------------------------------------------------- selection */

function currentSelection() {
  const doc = $('doc');
  const start = doc.selectionStart, end = doc.selectionEnd;
  if (start !== end) return { start, end, text: doc.value.slice(start, end), whole: false };
  return { start: 0, end: doc.value.length, text: doc.value, whole: true };
}

function refreshSelectionInfo() {
  const doc = $('doc');
  const chars = doc.selectionEnd - doc.selectionStart;
  const chip = $('sel-info');
  if (chars > 0) {
    const words = doc.value.slice(doc.selectionStart, doc.selectionEnd).trim().split(/\s+/).filter(Boolean).length;
    chip.textContent = `${chars} chars · ${words} words selected`;
    chip.className = 'chip chip-accent';
  } else {
    chip.textContent = doc.value.trim() ? 'nothing selected — edits apply to the whole document' : 'nothing selected';
    chip.className = 'chip';
  }
}

function replaceRange(start, end, text) {
  const doc = $('doc');
  state.undoStack.push({ value: doc.value, start: doc.selectionStart, end: doc.selectionEnd });
  $('btn-undo').disabled = false;
  doc.setRangeText(text, start, end, 'select');
  doc.focus();
  refreshSelectionInfo();
}

/* --------------------------------------------------------------- recording */

async function startRecording(mode) {
  if (state.busy) return;
  state.mode = mode;
  state.heard = '';
  state.selection = currentSelection();
  showInstruction(mode === 'dictate' ? 'dictating…' : 'listening…');
  $('instruction-text').value = '';
  resetSteps();
  setStep('record', 'active');

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    const why = {
      NotAllowedError: 'Microphone blocked — allow it for this site and try again.',
      NotFoundError: 'No microphone found.',
      NotReadableError: 'The microphone is in use by another application.',
    }[err.name];
    toast(why || `Microphone unavailable (${err.name}).`, true);
    resetSteps(); hideInstruction();
    return;
  }
  state.stream = stream;

  try {
    await openSocket(mode);
  } catch (err) {
    stopRecording(false);
    toast(err.message || 'Could not start the live connection.', true);
  }
}

async function openSocket(mode) {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/ws/live`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === 'delta') {
      state.heard += ev.text || '';
      $('instruction-text').value = state.heard.trim();
    } else if (ev.type === 'done') {
      state.asrMs = ev.stats ? Math.round(ev.stats.compute_seconds * 1000) : null;
      finishRecording(ev);
    } else if (ev.type === 'error') {
      toast(ev.message, true);
      stopRecording(false);
    }
  };
  ws.onerror = () => toast('Live connection failed.', true);

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    setTimeout(() => reject(new Error('the backend did not accept the live connection')), 8000);
  });

  // Dictation is verbatim; an instruction benefits from biasing toward the
  // words already on screen, which are exactly what it will talk about.
  const context = mode === 'edit' ? hotwordsFromDocument() : null;
  ws.send(JSON.stringify({ context_info: context, temperature: 0, max_new_tokens: 256 }));

  state.busy = true;
  const btn = mode === 'dictate' ? $('btn-dictate') : $('btn-edit');
  btn.classList.add('is-recording');
  btn.innerHTML = '<span class="rec-dot"></span> Stop';
  (mode === 'dictate' ? $('btn-edit') : $('btn-dictate')).disabled = true;

  const ctx = new AudioContext({ sampleRate: 24000 });
  state.audioCtx = ctx;
  await ctx.audioWorklet.addModule('pcm-processor.js');
  const source = ctx.createMediaStreamSource(state.stream);
  const node = new AudioWorkletNode(ctx, 'pcm-processor');
  state.worklet = node;
  node.port.onmessage = (e) => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(e.data);
  };
  const mute = ctx.createGain(); mute.gain.value = 0;
  source.connect(node); node.connect(mute).connect(ctx.destination);
}

/* Feed distinctive words from the document in as ASR hotwords, so an
   instruction that names a term already on screen gets it right. */
// Words an editing instruction is likely to contain. Biasing toward these
// costs nothing and cuts down the mishears that make you cancel in the first place.
const EDIT_VOCAB = [
  'rewrite', 'rephrase', 'replace', 'delete', 'shorten', 'expand', 'summarise',
  'bullet points', 'paragraph', 'sentence', 'formal', 'casual', 'concise',
  'past tense', 'present tense', 'capitalise', 'lowercase', 'spelling',
  'punctuation', 'heading', 'numbered list', 'instead of', 'change', 'fix',
];

function hotwordsFromDocument() {
  // Anything capitalised, hyphenated, numbered or snake_cased: proper nouns and
  // identifiers are what the ASR mangles. Ordinary sentence-initial words slip
  // through and are harmless -- they are already high-probability tokens.
  const words = ($('doc').value.match(/\b[A-Za-z][A-Za-z0-9_.-]{3,}\b/g) || [])
    .filter(w => /[A-Z]/.test(w) || /[0-9_.-]/.test(w));
  return [...new Set([...words.slice(0, 24), ...EDIT_VOCAB])].join(', ') || null;
}

function stopRecording(flush = true) {
  if (state.worklet) { try { state.worklet.port.onmessage = null; state.worklet.disconnect(); } catch (_) {} state.worklet = null; }
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  if (state.audioCtx) { state.audioCtx.close().catch(() => {}); state.audioCtx = null; }

  if (flush && state.ws && state.ws.readyState === WebSocket.OPEN) {
    setStep('record', 'done'); setStep('asr', 'active');
    state.ws.send('end');
  } else {
    if (state.ws) { try { state.ws.close(); } catch (_) {} state.ws = null; }
    resetControls();
    resetSteps();
  }
}

function resetControls() {
  state.busy = false;
  $('btn-dictate').disabled = false; $('btn-edit').disabled = false;
  $('btn-dictate').classList.remove('is-recording');
  $('btn-edit').classList.remove('is-recording');
  $('btn-dictate').innerHTML = '<span class="rec-dot"></span> Start dictating';
  $('btn-edit').innerHTML = '<span class="rec-dot"></span> Speak an instruction';
}

async function finishRecording(ev) {
  if (state.ws) { try { state.ws.close(); } catch (_) {} state.ws = null; }
  // `plain` has no speaker labels -- this is one person talking, and a
  // "Speaker 0:" prefix would land in the document or the LLM instruction.
  const heard = (ev.plain || state.heard || '').trim();
  setStep('record', 'done'); setStep('asr', 'done');
  $('m-asr').textContent = state.asrMs != null ? state.asrMs + ' ms' : '—';

  if (!heard) { toast('Nothing was transcribed.', true); resetControls(); resetSteps(); hideInstruction(); return; }

  if (state.mode === 'dictate') {
    const sel = state.selection;
    // Dictation goes in at the cursor; a live selection is overwritten.
    const doc = $('doc');
    const start = sel.whole ? doc.selectionStart : sel.start;
    const end = sel.whole ? doc.selectionEnd : sel.end;
    replaceRange(start, end, heard);
    setStep('apply', 'done');
    resetControls();
    hideInstruction();
    addHistory('Dictated', heard, `${state.asrMs ?? '?'} ms`);
    return;
  }

  $('instruction-text').value = heard;
  if ($('review-mode').checked) {
    // Hold here so a mis-heard instruction can be corrected or thrown away
    // before it costs an API call and a wrong edit.
    state.awaitingReview = true;
    resetControls();
    showInstruction('instruction — check it', {
      editable: true,
      hint: 'edit it, <kbd>&crarr;</kbd> to apply, <kbd>Esc</kbd> to discard',
    });
    return;
  }
  await runEdit(heard);
}

/* --------------------------------------------------------------- the edit */

async function runEdit(instruction) {
  const sel = state.selection;
  if (!sel.text.trim()) { toast('There is no text to edit.', true); resetControls(); resetSteps(); return; }

  state.awaitingReview = false;
  state.lastInstruction = instruction;
  state.lastSelection = sel;
  showInstruction('editing…', { hint: '<kbd>Esc</kbd> cancels before anything is replaced' });
  $('instruction-text').value = instruction;
  setStep('llm', 'active');
  const body = {
    text: sel.text,
    instruction,
    model: $('model').value || undefined,
    system_prompt: $('system-prompt').value,
    temperature: parseFloat($('temperature').value),
  };
  const key = $('api-key').value.trim();
  if (key) body.api_key = key;

  const controller = new AbortController();
  state.editAbort = controller;
  try {
    const res = await fetch('/api/llm/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);

    setStep('llm', 'done'); setStep('apply', 'active');
    const before = sel.text;
    const after = payload.result;
    replaceRange(sel.start, sel.start + before.length, after);
    setStep('apply', 'done');

    $('m-llm').textContent = Math.round(payload.elapsed_ms) + ' ms';
    $('m-cost').textContent = formatCost(payload);
    $('result').hidden = false;
    $('result-meta').textContent = `${payload.model} · ${payload.usage.total_tokens ?? '?'} tokens`;
    $('diff').innerHTML = diffWords(before, after);
    addHistory(instruction, after, `${Math.round(payload.elapsed_ms)} ms`);
  } catch (err) {
    setStep('llm', '');
    // An abort is a deliberate cancel, not a failure -- cancelAll already said so.
    if (err.name !== 'AbortError') toast(err.message, true);
  } finally {
    state.editAbort = null;
    resetControls();
    if (!state.awaitingReview) hideInstruction();
  }
}

function formatCost(payload) {
  const model = state.models.find(m => m.id === payload.model) ||
                state.models.find(m => m.id === $('model').value);
  const usage = payload.usage || {};
  if (!model || model.prompt_price == null || usage.prompt_tokens == null) return '—';
  const cost = usage.prompt_tokens * model.prompt_price +
               (usage.completion_tokens || 0) * (model.completion_price || 0);
  if (cost < 0.01) return '<$0.01';
  return '$' + cost.toFixed(3);
}

function addHistory(label, text, timing) {
  const box = $('history');
  if (box.querySelector('.hint')) box.innerHTML = '';
  const item = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `<b>${escapeHtml(label.slice(0, 90))}</b><span>${escapeHtml(text.slice(0, 140))}</span><em>${timing}</em>`;
  box.prepend(item);
}

/* ------------------------------------------------------------------ setup */

async function loadStatus() {
  try {
    const s = await (await fetch('/api/llm/status')).json();
    state.serverKey = s.server_key;
    state.defaultPrompt = s.default_system_prompt;
    $('key-server').hidden = !s.server_key;
    $('key-field').hidden = s.server_key;
    if (!$('system-prompt').value) $('system-prompt').value = localStorage.getItem(LS.prompt) || s.default_system_prompt;
  } catch (_) { /* health poll reports the outage */ }
}

async function loadModels() {
  try {
    const models = await (await fetch('/api/llm/models')).json();
    state.models = models;
    renderModels();
  } catch (err) {
    $('model-count').textContent = 'unavailable';
  }
}

function renderModels() {
  const filter = $('model-search').value.trim().toLowerCase();
  const wanted = localStorage.getItem(LS.model) || 'anthropic/claude-haiku-4.5';
  const list = state.models.filter(m => !filter || m.id.toLowerCase().includes(filter) || (m.name || '').toLowerCase().includes(filter));
  const select = $('model');
  select.innerHTML = '';
  for (const m of list.slice(0, 400)) {
    const option = document.createElement('option');
    option.value = m.id;
    option.textContent = (m.suggested ? '★ ' : '') + m.id;
    if (m.id === wanted) option.selected = true;
    select.appendChild(option);
  }
  if (!select.value && select.options.length) select.options[0].selected = true;
  $('model-count').textContent = `${list.length} of ${state.models.length}`;
  showPrice();
}

function showPrice() {
  const m = state.models.find(x => x.id === $('model').value);
  if (!m) { $('model-price').textContent = '—'; return; }
  const per = (p) => p == null ? '?' : '$' + (p * 1e6).toFixed(2);
  $('model-price').textContent = `${per(m.prompt_price)} in / ${per(m.completion_price)} out per 1M tokens · ${(m.context_length || 0).toLocaleString()} ctx`;
}

async function pollHealth() {
  try {
    const info = await (await fetch('/api/health')).json();
    const dot = $('health-dot'), text = $('health-text');
    if (info.status === 'ready') { dot.className = 'dot ok'; text.textContent = 'ASR ready'; return true; }
    dot.className = info.status === 'loading' ? 'dot busy' : 'dot err';
    text.textContent = info.status === 'loading' ? 'loading model…' : 'backend error';
  } catch (_) {
    $('health-dot').className = 'dot err';
    $('health-text').textContent = 'backend unreachable';
  }
  return false;
}

function scheduleHealth(delay) {
  setTimeout(async () => scheduleHealth(await pollHealth() ? 15000 : 3000), delay);
}

function init() {
  $('doc').value = SAMPLE;

  const savedKey = localStorage.getItem(LS.key);
  if (savedKey) $('api-key').value = savedKey;
  $('api-key').addEventListener('change', (e) => localStorage.setItem(LS.key, e.target.value.trim()));

  $('model-search').addEventListener('input', renderModels);
  $('model').addEventListener('change', () => { localStorage.setItem(LS.model, $('model').value); showPrice(); });
  $('temperature').addEventListener('input', (e) => { $('temp-val').textContent = parseFloat(e.target.value).toFixed(2); });
  $('system-prompt').addEventListener('change', (e) => localStorage.setItem(LS.prompt, e.target.value));
  $('btn-reset-prompt').addEventListener('click', () => {
    $('system-prompt').value = state.defaultPrompt;
    localStorage.removeItem(LS.prompt);
    toast('System prompt reset.');
  });

  ['select', 'keyup', 'mouseup', 'input', 'focus'].forEach(ev =>
    $('doc').addEventListener(ev, refreshSelectionInfo));
  refreshSelectionInfo();

  const toggle = (mode) => () => (state.busy && state.mode === mode) ? stopRecording(true) : startRecording(mode);
  $('btn-dictate').addEventListener('click', toggle('dictate'));
  $('btn-edit').addEventListener('click', toggle('edit'));

  const applyReviewed = () => {
    const instruction = $('instruction-text').value.trim();
    if (!instruction) { toast('The instruction is empty.', true); return; }
    runEdit(instruction);
  };
  $('btn-apply').addEventListener('click', applyReviewed);
  $('btn-discard').addEventListener('click', () => cancelAll('Discarded — nothing changed.'));
  $('btn-rerecord').addEventListener('click', () => {
    // Keep the same target text; just take the instruction again.
    const sel = state.selection;
    hideInstruction();
    startRecording('edit').then(() => { if (sel) state.selection = sel; });
  });

  $('instruction-text').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && state.awaitingReview) { e.preventDefault(); applyReviewed(); }
  });

  $('btn-retry').addEventListener('click', () => {
    if (!state.lastInstruction || !state.lastSelection) return;
    $('btn-undo').click();                       // put the original text back
    state.selection = state.lastSelection;
    runEdit(state.lastInstruction);
  });

  $('review-mode').addEventListener('change', (e) =>
    localStorage.setItem(LS.review, e.target.checked ? '1' : '0'));
  const savedReview = localStorage.getItem(LS.review);
  if (savedReview !== null) $('review-mode').checked = savedReview === '1';

  // In-page stand-ins for the eventual global hotkeys.
  window.addEventListener('keydown', (e) => {
    // Esc means the same thing at every stage: stop, change nothing.
    if (e.key === 'Escape' && inFlight()) {
      e.preventDefault();
      cancelAll();
      return;
    }
    if (!e.ctrlKey || !e.shiftKey) return;
    const k = e.key.toLowerCase();
    if (k === 'd') { e.preventDefault(); toggle('dictate')(); }
    if (k === 'e') { e.preventDefault(); toggle('edit')(); }
  });

  $('btn-undo').addEventListener('click', () => {
    const prev = state.undoStack.pop();
    if (!prev) return;
    $('doc').value = prev.value;
    $('doc').setSelectionRange(prev.start, prev.end);
    $('doc').focus();
    $('btn-undo').disabled = !state.undoStack.length;
    refreshSelectionInfo();
  });
  $('btn-sample').addEventListener('click', () => {
    state.undoStack.push({ value: $('doc').value, start: 0, end: 0 });
    $('btn-undo').disabled = false;
    $('doc').value = SAMPLE; refreshSelectionInfo();
  });
  $('btn-clear-doc').addEventListener('click', () => {
    state.undoStack.push({ value: $('doc').value, start: 0, end: 0 });
    $('btn-undo').disabled = false;
    $('doc').value = ''; refreshSelectionInfo();
  });

  loadStatus();
  loadModels();
  scheduleHealth(0);
}

document.addEventListener('DOMContentLoaded', init);
