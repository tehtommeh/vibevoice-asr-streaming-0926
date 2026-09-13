/* Voice editing: one key. Something selected -> speech is an instruction about
   it. Nothing selected -> speech is dictation at the cursor.

   A browser stand-in for the desktop hotkey, so the pipeline, the prompts and
   the command vocabulary can be settled before any of it touches /dev/uinput. */
'use strict';

const $ = (id) => document.getElementById(id);
const LS = {
  key: 'vv.openrouter.key', model: 'vv.openrouter.model',
  review: 'vv.review', grace: 'vv.grace', cleanup: 'vv.cleanup',
};
const TAP_MS = 350;        // press shorter than this latches instead of push-to-talk

const SAMPLE = `We deployed the new service to cooper netties last Tuesday and it's been running fine since. The main thing that changed is we moved the ingest path off of the old queue and onto red is streams, which cut the tail latency by about a third.

There are still two open questions. First, whether we keep the fallback path around at all. Second, how we want to handle back pressure when the consumer falls behind.`;

const state = {
  busy: false, mode: null, ws: null, audioCtx: null, worklet: null, stream: null,
  heard: '', models: [], serverKey: false,
  modes: [], activeMode: null, commands: [], vocab: {}, hotwords: [],
  selection: null, undoStack: [], asrMs: null,
  awaitingReview: false, editAbort: null, lastInstruction: null, lastSelection: null,
  countdownTimer: null, pressAt: 0, latched: false,
};

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
function resetSteps() { ['record', 'asr', 'llm', 'apply'].forEach(s => setStep(s, '')); }

function escapeHtml(t) {
  return t.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* Word-level LCS diff. Returns runs so the same pass can render the diff and
   feed the vocabulary learner. */
function diffRuns(before, after) {
  const a = before.split(/(\s+)/), b = after.split(/(\s+)/);
  const n = a.length, m = b.length;
  if (n * m > 4000000) return [{ type: 'del', text: before }, { type: 'ins', text: after }];

  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);

  const runs = [];
  const push = (type, text) => {
    if (!text) return;
    const last = runs[runs.length - 1];
    if (last && last.type === type) last.text += text; else runs.push({ type, text });
  };
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { push('same', a[i]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) push('del', a[i++]);
    else push('ins', b[j++]);
  }
  while (i < n) push('del', a[i++]);
  while (j < m) push('ins', b[j++]);
  return runs;
}

function renderDiff(runs) {
  return runs.map(r => r.type === 'same' ? escapeHtml(r.text)
    : `<${r.type === 'del' ? 'del' : 'ins'}>${escapeHtml(r.text)}</${r.type === 'del' ? 'del' : 'ins'}>`).join('');
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
  if (editable) {
    // Focus next frame: the control labels are rewritten just before this, which
    // drops focus to <body> if we grab it too early.
    requestAnimationFrame(() => { box.focus(); box.setSelectionRange(box.value.length, box.value.length); });
  }
  $('instruction').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function hideInstruction() {
  stopCountdown();
  $('instruction').hidden = true;
  $('instruction-text').value = '';
  $('instruction-actions').hidden = true;
  state.awaitingReview = false;
}

/* The grace period always resolves on its own -- a step that waits forever for
   a keypress reads as a hang. */
function startCountdown(onElapsed) {
  stopCountdown();
  const ms = Math.round(parseFloat($('grace').value) * 1000);
  const fill = $('countdown-fill');
  $('countdown').hidden = false;
  fill.style.transition = 'none';
  fill.style.width = '100%';
  void fill.offsetWidth;
  fill.style.transition = `width ${ms}ms linear`;
  fill.style.width = '0%';
  state.countdownTimer = setTimeout(() => { state.countdownTimer = null; onElapsed(); }, ms);
}

function stopCountdown() {
  if (state.countdownTimer) { clearTimeout(state.countdownTimer); state.countdownTimer = null; }
  const fill = $('countdown-fill');
  fill.style.transition = 'none';
  fill.style.width = '0%';
  $('countdown').hidden = true;
}

function cancelAll(reason) {
  const wasDoing = inFlight();
  stopCountdown();
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
  return { start, end, text: '', whole: true, caret: start };
}

function refreshSelectionInfo() {
  const doc = $('doc');
  const chars = doc.selectionEnd - doc.selectionStart;
  const chip = $('sel-info');
  const hasSel = chars > 0;
  if (hasSel) {
    const words = doc.value.slice(doc.selectionStart, doc.selectionEnd).trim().split(/\s+/).filter(Boolean).length;
    chip.textContent = `${chars} chars · ${words} words selected`;
    chip.className = 'chip chip-accent';
  } else {
    chip.textContent = 'nothing selected';
    chip.className = 'chip';
  }
  if (!state.busy) {
    $('talk-label').textContent = hasSel ? 'Hold to edit selection' : 'Hold to dictate';
    $('talk-target').textContent = hasSel ? 'speech = an instruction about the selection'
                                          : 'speech = text at the cursor';
  }
  $('btn-talk').classList.toggle('is-edit', hasSel);
}

function replaceRange(start, end, text) {
  const doc = $('doc');
  state.undoStack.push({ value: doc.value, start: doc.selectionStart, end: doc.selectionEnd });
  $('btn-undo').disabled = false;
  doc.setRangeText(text, start, end, 'select');
  doc.focus();
  refreshSelectionInfo();
}

/* --------------------------------------------------------- voice commands */

function normalise(text) {
  return text.toLowerCase().replace(/[.,!?;:'"]/g, '').replace(/\s+/g, ' ').trim();
}

/* Returns an action when the whole utterance is a command, or when it trails
   off into a cancellation ("make it bold, no, never mind"). */
function matchCommand(text) {
  const norm = normalise(text);
  if (!norm) return null;
  for (const cmd of state.commands) {
    for (const phrase of cmd.phrases) {
      if (norm === normalise(phrase)) return { action: cmd.action, phrase, whole: true };
    }
  }
  const cancel = state.commands.find(c => c.action === 'cancel');
  if (cancel) {
    for (const phrase of cancel.phrases) {
      const p = normalise(phrase);
      if (norm.endsWith(' ' + p) || norm === p) return { action: 'cancel', phrase, whole: false };
    }
  }
  return null;
}

const TRANSFORMS = {
  uppercase: (t) => t.toUpperCase(),
  lowercase: (t) => t.toLowerCase(),
  titlecase: (t) => t.replace(/\w\S*/g, w => w[0].toUpperCase() + w.slice(1).toLowerCase()),
  sentencecase: (t) => t.charAt(0).toUpperCase() + t.slice(1).toLowerCase(),
  trim: (t) => t.replace(/\s+/g, ' ').trim(),
};

/* Run a command locally. Returns true if it was handled. */
function runCommand(action, sel) {
  const doc = $('doc');
  if (action === 'cancel') { cancelAll('Cancelled — nothing changed.'); return true; }
  if (action === 'undo') {
    if ($('btn-undo').disabled) { toast('Nothing to undo.'); } else { $('btn-undo').click(); toast('Undone.'); }
    return true;
  }
  if (action === 'newline' || action === 'paragraph') {
    const text = action === 'paragraph' ? '\n\n' : '\n';
    replaceRange(sel.start, sel.end, text);
    addHistory(action === 'paragraph' ? 'new paragraph' : 'new line', '', 'local · 0 ms');
    return true;
  }
  if (!sel.text) { toast('Select some text for that one.', true); return true; }
  if (action === 'delete') {
    replaceRange(sel.start, sel.end, '');
    addHistory('delete that', '', 'local · 0 ms');
    return true;
  }
  const fn = TRANSFORMS[action];
  if (fn) {
    const after = fn(sel.text);
    replaceRange(sel.start, sel.end, after);
    showResult(sel.text, after, { local: true, label: action });
    addHistory(action, after, 'local · 0 ms');
    return true;
  }
  return false;
}

/* --------------------------------------------------------------- recording */

function talkTargetIsEdit() {
  const doc = $('doc');
  return doc.selectionEnd > doc.selectionStart;
}

async function startRecording() {
  if (state.busy) return;
  state.selection = currentSelection();
  state.mode = talkTargetIsEdit() ? 'edit' : 'dictate';
  state.heard = '';
  showInstruction(state.mode === 'dictate' ? 'dictating…' : 'listening…');
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
    resetSteps(); hideInstruction(); resetControls();
    return;
  }
  state.stream = stream;
  try {
    await openSocket();
  } catch (err) {
    stopRecording(false);
    toast(err.message || 'Could not start the live connection.', true);
  }
}

async function openSocket() {
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
  ws.send(JSON.stringify({ context_info: asrHints(), temperature: 0, max_new_tokens: 256 }));

  state.busy = true;
  $('btn-talk').classList.add('is-recording');
  $('talk-label').textContent = state.latched ? 'Recording — tap to stop' : 'Listening… release to stop';

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

// Words an editing instruction is likely to contain, plus terms already on
// screen and everything the user has taught us.
const EDIT_VOCAB = [
  'rewrite', 'rephrase', 'replace', 'delete', 'shorten', 'expand', 'summarise',
  'bullet points', 'paragraph', 'sentence', 'formal', 'casual', 'concise',
  'past tense', 'present tense', 'capitalise', 'lowercase', 'spelling',
  'punctuation', 'heading', 'numbered list', 'instead of', 'never mind',
  'scratch that', 'undo', 'change', 'fix',
];

function asrHints() {
  const words = ($('doc').value.match(/\b[A-Za-z][A-Za-z0-9_.-]{3,}\b/g) || [])
    .filter(w => /[A-Z]/.test(w) || /[0-9_.-]/.test(w));
  const all = [...new Set([...state.hotwords, ...words.slice(0, 24), ...EDIT_VOCAB])];
  return all.join(', ') || null;
}

function stopRecording(flush = true) {
  if (state.worklet) { try { state.worklet.port.onmessage = null; state.worklet.disconnect(); } catch (_) {} state.worklet = null; }
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  if (state.audioCtx) { state.audioCtx.close().catch(() => {}); state.audioCtx = null; }
  state.latched = false;

  if (flush && state.ws && state.ws.readyState === WebSocket.OPEN) {
    setStep('record', 'done'); setStep('asr', 'active');
    $('talk-label').textContent = 'Transcribing…';
    state.ws.send('end');
  } else {
    if (state.ws) { try { state.ws.close(); } catch (_) {} state.ws = null; }
    resetControls();
    resetSteps();
  }
}

function resetControls() {
  state.busy = false;
  state.latched = false;
  $('btn-talk').classList.remove('is-recording');
  refreshSelectionInfo();
}

async function finishRecording(ev) {
  if (state.ws) { try { state.ws.close(); } catch (_) {} state.ws = null; }
  // `plain` has no speaker labels -- one person talking, and "Speaker 0:"
  // would land in the document or the instruction.
  const heard = (ev.plain || state.heard || '').trim();
  setStep('record', 'done'); setStep('asr', 'done');
  $('m-asr').textContent = state.asrMs != null ? state.asrMs + ' ms' : '—';

  if (!heard) { toast('Nothing was transcribed.', true); resetControls(); resetSteps(); hideInstruction(); return; }

  // Spoken commands short-circuit the whole pipeline: no API call, no latency.
  const command = matchCommand(heard);
  if (command) {
    resetControls();
    hideInstruction();
    if (command.action === 'cancel') {
      resetSteps();
      toast(`“${command.phrase}” — cancelled, nothing changed.`);
      return;
    }
    if (runCommand(command.action, state.selection)) { setStep('apply', 'done'); return; }
  }

  if (state.mode === 'dictate') return finishDictation(heard);

  $('instruction-text').value = heard;
  if ($('review-mode').checked) {
    state.awaitingReview = true;
    resetControls();
    showInstruction('heard — applying shortly', {
      editable: true,
      hint: '<kbd>&crarr;</kbd> now · <kbd>Esc</kbd> cancel · type to hold',
    });
    startCountdown(() => {
      if (!state.awaitingReview) return;
      runEdit($('instruction-text').value.trim());
    });
    return;
  }
  await runEdit(heard);
}

async function finishDictation(heard) {
  const sel = state.selection;
  let text = heard;
  if ($('cleanup-dictation').checked) {
    setStep('llm', 'active');
    showInstruction('cleaning up…', { hint: '<kbd>Esc</kbd> cancels' });
    const cleaned = await callLLM({ text: heard, instruction: '', task: 'dictate' });
    if (cleaned === null) return;                 // cancelled or failed
    text = cleaned.result || heard;
    $('m-llm').textContent = Math.round(cleaned.elapsed_ms) + ' ms';
    $('m-cost').textContent = formatCost(cleaned);
    setStep('llm', 'done');
    if (text !== heard) showResult(heard, text, { label: 'dictation cleanup' });
  }
  replaceRange(sel.start, sel.end, text);
  setStep('apply', 'done');
  resetControls();
  hideInstruction();
  addHistory('Dictated', text, `${state.asrMs ?? '?'} ms`);
}

/* --------------------------------------------------------------- the edit */

async function callLLM({ text, instruction, task = 'edit' }) {
  const body = {
    text, instruction, task,
    mode: state.activeMode,
    model: $('model').value || undefined,
    temperature: parseFloat($('temperature').value),
  };
  if (task === 'edit') body.system_prompt = $('system-prompt').value;
  else body.system_prompt = $('dictation-prompt').value;
  const key = $('api-key').value.trim();
  if (key) body.api_key = key;

  const controller = new AbortController();
  state.editAbort = controller;
  try {
    const res = await fetch('/api/llm/edit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: controller.signal,
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
    return payload;
  } catch (err) {
    setStep('llm', '');
    if (err.name !== 'AbortError') toast(err.message, true);
    resetControls();
    hideInstruction();
    return null;
  } finally {
    state.editAbort = null;
  }
}

async function runEdit(instruction) {
  const sel = state.selection;
  const target = sel.text || $('doc').value;
  if (!target.trim()) { toast('There is no text to edit.', true); resetControls(); resetSteps(); hideInstruction(); return; }

  state.awaitingReview = false;
  stopCountdown();
  state.lastInstruction = instruction;
  state.lastSelection = sel;
  showInstruction('editing…', { hint: '<kbd>Esc</kbd> cancels before anything is replaced' });
  $('instruction-text').value = instruction;
  setStep('llm', 'active');

  const range = sel.text ? [sel.start, sel.end] : [0, $('doc').value.length];
  const payload = await callLLM({ text: target, instruction });
  if (payload === null) return;

  setStep('llm', 'done'); setStep('apply', 'active');
  const after = payload.result;
  replaceRange(range[0], range[1], after);
  setStep('apply', 'done');

  $('m-llm').textContent = Math.round(payload.elapsed_ms) + ' ms';
  $('m-cost').textContent = formatCost(payload);
  const runs = showResult(target, after, { meta: `${payload.model} · ${payload.usage.total_tokens ?? '?'} tokens` });
  learnFromDiff(runs, instruction);
  addHistory(instruction, after, `${Math.round(payload.elapsed_ms)} ms`);
  resetControls();
  hideInstruction();
}

function showResult(before, after, { meta = '', local = false, label = '' } = {}) {
  const runs = diffRuns(before, after);
  $('result').hidden = false;
  $('result-meta').textContent = meta || (local ? `local command · ${label}` : label);
  $('diff').innerHTML = renderDiff(runs);
  return runs;
}

/* Terms the model introduced that the recogniser got wrong are exactly the
   words worth biasing toward next time. */
function learnFromDiff(runs, instruction) {
  const learned = [];
  for (let i = 0; i < runs.length; i++) {
    if (runs[i].type !== 'ins') continue;
    const removed = (i > 0 && runs[i - 1].type === 'del') ? runs[i - 1].text.trim() : '';
    for (const word of runs[i].text.match(/\b[A-Za-z][A-Za-z0-9_.-]{2,}\b/g) || []) {
      if (!/[A-Z]/.test(word) && !/[0-9_.-]/.test(word)) continue;   // ordinary word
      if (new RegExp(`\\b${word}\\b`, 'i').test(instruction) || removed) {
        learned.push({ term: word, heard: removed.slice(0, 80) || undefined });
      }
    }
  }
  if (!learned.length) return;
  fetch('/api/vocab', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ terms: learned.slice(0, 8) }),
  }).then(r => r.json()).then(d => { state.vocab = d.vocab; state.hotwords = d.hotwords; renderVocab(); })
    .catch(() => {});
}

function formatCost(payload) {
  const model = state.models.find(m => m.id === payload.model) ||
                state.models.find(m => m.id === $('model').value);
  const usage = payload.usage || {};
  if (!model || model.prompt_price == null || usage.prompt_tokens == null) return '—';
  const cost = usage.prompt_tokens * model.prompt_price +
               (usage.completion_tokens || 0) * (model.completion_price || 0);
  return cost < 0.01 ? '<$0.01' : '$' + cost.toFixed(3);
}

function addHistory(label, text, timing) {
  const box = $('history');
  if (box.querySelector('.hint')) box.innerHTML = '';
  const item = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `<b>${escapeHtml(String(label).slice(0, 90))}</b>` +
                   (text ? `<span>${escapeHtml(text.slice(0, 140))}</span>` : '') +
                   `<em>${timing}</em>`;
  box.prepend(item);
}

/* ------------------------------------------------------- modes and vocab */

function applyMode(id) {
  const mode = state.modes.find(m => m.id === id) || state.modes[0];
  if (!mode) return;
  state.activeMode = mode.id;
  $('mode').value = mode.id;
  $('system-prompt').value = mode.system_prompt || '';
  $('dictation-prompt').value = mode.dictation_prompt || '';
  $('temperature').value = mode.temperature ?? 0.2;
  $('temp-val').textContent = parseFloat($('temperature').value).toFixed(2);
  if (mode.model) { $('model').value = mode.model; showPrice(); }
  fetch('/api/modes/active', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: mode.id }),
  }).catch(() => {});
}

async function loadModes() {
  try {
    const d = await (await fetch('/api/modes')).json();
    state.modes = d.modes;
    $('mode').innerHTML = d.modes.map(m => `<option value="${m.id}">${escapeHtml(m.name)}</option>`).join('');
    applyMode(d.active);
  } catch (_) { /* health poll reports the outage */ }
}

async function saveMode() {
  const mode = state.modes.find(m => m.id === state.activeMode);
  if (!mode) return;
  mode.system_prompt = $('system-prompt').value;
  mode.dictation_prompt = $('dictation-prompt').value;
  mode.temperature = parseFloat($('temperature').value);
  mode.model = $('model').value || '';
  try {
    const res = await fetch('/api/modes', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modes: state.modes, active: state.activeMode }),
    });
    if (!res.ok) throw new Error('save failed');
    toast(`Saved “${mode.name}”.`);
  } catch (err) { toast(err.message, true); }
}

async function loadVocab() {
  try {
    const d = await (await fetch('/api/vocab')).json();
    state.vocab = d.vocab; state.hotwords = d.hotwords;
    $('promote-at').textContent = d.promote_at;
    renderVocab();
  } catch (_) {}
}

function renderVocab() {
  const box = $('vocab-list');
  const entries = Object.entries(state.vocab).sort((a, b) => b[1].count - a[1].count);
  if (!entries.length) { box.innerHTML = '<p class="hint">Nothing learned yet.</p>'; return; }
  box.innerHTML = '';
  for (const [term, info] of entries) {
    const row = document.createElement('div');
    row.className = 'vocab-item' + (state.hotwords.includes(term) ? ' promoted' : '');
    row.innerHTML = `<b>${escapeHtml(term)}</b>` +
      (info.heard && info.heard.length ? `<span>heard as “${escapeHtml(info.heard[0])}”</span>` : '') +
      `<em>×${info.count}</em>`;
    const del = document.createElement('button');
    del.className = 'vocab-del'; del.textContent = '×'; del.title = 'Forget this term';
    del.addEventListener('click', async () => {
      const d = await (await fetch('/api/vocab/' + encodeURIComponent(term), { method: 'DELETE' })).json();
      state.vocab = d.vocab; state.hotwords = d.hotwords; renderVocab();
    });
    row.appendChild(del);
    box.appendChild(row);
  }
}

async function loadCommands() {
  try {
    state.commands = await (await fetch('/api/commands')).json();
    $('command-list').innerHTML = state.commands
      .map(c => `<div class="cmd"><b>${escapeHtml(c.phrases[0])}</b><span>${escapeHtml(c.action)}</span></div>`)
      .join('');
  } catch (_) { state.commands = []; }
}

/* -------------------------------------------------------- models / health */

async function loadStatus() {
  try {
    const s = await (await fetch('/api/llm/status')).json();
    state.serverKey = s.server_key;
    $('key-server').hidden = !s.server_key;
    $('key-field').hidden = s.server_key;
  } catch (_) {}
}

async function loadModels() {
  try { state.models = await (await fetch('/api/llm/models')).json(); renderModels(); }
  catch (_) { $('model-count').textContent = 'unavailable'; }
}

function renderModels() {
  const filter = $('model-search').value.trim().toLowerCase();
  const wanted = $('model').value || localStorage.getItem(LS.model) || 'anthropic/claude-haiku-4.5';
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
  $('model-price').textContent =
    `${per(m.prompt_price)} in / ${per(m.completion_price)} out per 1M · ${(m.context_length || 0).toLocaleString()} ctx`;
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

/* --------------------------------------------------- press-to-talk wiring */

function pressStart() {
  if (inFlight()) return;
  state.pressAt = Date.now();
  state.latched = false;
  startRecording();
}

function pressEnd() {
  if (!state.busy) return;
  if (Date.now() - state.pressAt < TAP_MS) {
    // Too short to be a deliberate hold: latch on, stop on the next press.
    state.latched = true;
    $('talk-label').textContent = 'Recording — tap to stop';
    return;
  }
  stopRecording(true);
}

function talkPressed() {
  if (state.latched && state.busy) { stopRecording(true); return true; }
  return false;
}

function init() {
  $('doc').value = SAMPLE;

  const savedKey = localStorage.getItem(LS.key);
  if (savedKey) $('api-key').value = savedKey;
  $('api-key').addEventListener('change', e => localStorage.setItem(LS.key, e.target.value.trim()));

  $('model-search').addEventListener('input', renderModels);
  $('model').addEventListener('change', () => { localStorage.setItem(LS.model, $('model').value); showPrice(); });
  $('temperature').addEventListener('input', e => { $('temp-val').textContent = parseFloat(e.target.value).toFixed(2); });
  $('mode').addEventListener('change', e => applyMode(e.target.value));
  $('btn-save-mode').addEventListener('click', saveMode);
  $('btn-reset-modes').addEventListener('click', async () => {
    if (!confirm('Reset every mode to its shipped prompts? Learned vocabulary is kept.')) return;
    try {
      const res = await fetch('/api/modes/reset', { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadModes();
      toast('Modes reset to defaults.');
    } catch (err) { toast(err.message, true); }
  });

  ['select', 'keyup', 'mouseup', 'input', 'focus'].forEach(ev =>
    $('doc').addEventListener(ev, refreshSelectionInfo));
  refreshSelectionInfo();

  const talk = $('btn-talk');
  talk.addEventListener('mousedown', (e) => { e.preventDefault(); if (!talkPressed()) pressStart(); });
  talk.addEventListener('mouseup', pressEnd);
  talk.addEventListener('mouseleave', () => { if (state.busy && !state.latched) pressEnd(); });

  const applyReviewed = () => {
    const instruction = $('instruction-text').value.trim();
    if (!instruction) { toast('The instruction is empty.', true); return; }
    runEdit(instruction);
  };
  $('btn-apply').addEventListener('click', applyReviewed);
  $('btn-discard').addEventListener('click', () => cancelAll('Discarded — nothing changed.'));
  $('btn-rerecord').addEventListener('click', () => {
    const sel = state.selection;
    hideInstruction();
    startRecording().then(() => { if (sel) state.selection = sel; });
  });
  $('instruction-text').addEventListener('input', () => {
    if (state.countdownTimer) {
      stopCountdown();
      $('instruction-label').textContent = 'heard — edited, press Enter';
      $('instruction-hint').innerHTML = '<kbd>&crarr;</kbd> to apply · <kbd>Esc</kbd> to cancel';
    }
  });

  $('btn-retry').addEventListener('click', () => {
    if (!state.lastInstruction || !state.lastSelection) return;
    $('btn-undo').click();
    state.selection = state.lastSelection;
    runEdit(state.lastInstruction);
  });

  $('btn-vocab-add').addEventListener('click', async () => {
    const term = $('vocab-input').value.trim();
    if (!term) return;
    for (let i = 0; i < 2; i++) {                      // straight to promoted
      await fetch('/api/vocab', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ term }) });
    }
    $('vocab-input').value = '';
    loadVocab();
  });
  $('vocab-input').addEventListener('keydown', e => { if (e.key === 'Enter') $('btn-vocab-add').click(); });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && inFlight()) { e.preventDefault(); cancelAll(); return; }
    if (e.key === 'Enter' && !e.shiftKey && state.awaitingReview && e.target.id !== 'doc') {
      e.preventDefault(); applyReviewed(); return;
    }
    if (e.ctrlKey && e.code === 'Space') {
      e.preventDefault();
      if (e.repeat) return;                            // auto-repeat is not a new press
      if (!talkPressed()) pressStart();
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' || e.key === 'Control') { if (state.busy) pressEnd(); }
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

  const savedReview = localStorage.getItem(LS.review);
  if (savedReview !== null) $('review-mode').checked = savedReview === '1';
  $('review-mode').addEventListener('change', e => localStorage.setItem(LS.review, e.target.checked ? '1' : '0'));

  const savedCleanup = localStorage.getItem(LS.cleanup);
  if (savedCleanup !== null) $('cleanup-dictation').checked = savedCleanup === '1';
  $('cleanup-dictation').addEventListener('change', e => localStorage.setItem(LS.cleanup, e.target.checked ? '1' : '0'));

  const showGrace = () => { $('grace-val').textContent = parseFloat($('grace').value).toFixed(1) + 's'; };
  $('grace').addEventListener('input', () => { showGrace(); localStorage.setItem(LS.grace, $('grace').value); });
  const savedGrace = localStorage.getItem(LS.grace);
  if (savedGrace !== null) $('grace').value = savedGrace;
  showGrace();

  loadStatus(); loadModels(); loadModes(); loadVocab(); loadCommands();
  scheduleHealth(0);
}

document.addEventListener('DOMContentLoaded', init);
