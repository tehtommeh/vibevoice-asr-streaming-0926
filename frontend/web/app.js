/* VibeVoice Streaming ASR demo — talks to the FastAPI backend through nginx. */
'use strict';

const $ = (id) => document.getElementById(id);
const SPEAKER_COLORS = ['#6d8cff', '#3ddc97', '#ffb454', '#ff7ac6', '#5ad1ff', '#c79bff', '#ff9d6b', '#9ee85f'];

const state = {
  mode: 'mic',
  config: null,
  chunkSeconds: 2.93,
  source: null,          // { name, blob, url, hotwords }
  samples: [],
  busy: false,
  abort: null,
  ws: null,
  audioCtx: null,
  workletNode: null,
  micStream: null,
  recordStart: 0,
  timerId: null,
  latencies: [],
  levels: new Array(140).fill(0),
};

/* ------------------------------------------------------------------ utils */

function toast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.className = 'toast' + (isError ? ' err' : '');
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 2800);
}

function clock(seconds) {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function setStatus(text, cls = '') {
  const el = $('stage-status');
  el.textContent = text;
  el.className = 'chip' + (cls ? ' ' + cls : '');
}

async function* sseEvents(response, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split;
      while ((split = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          try { yield JSON.parse(line.slice(5).trim()); } catch (_) { /* keep-alive */ }
        }
      }
    }
  } finally {
    if (signal && signal.aborted) { try { await reader.cancel(); } catch (_) {} }
  }
}

/* ------------------------------------------------------------- transcript */

const transcript = {
  nodes: new Map(),
  speakers: new Map(),
  segments: [],
  raw: '',
  liveNode: null,

  reset() {
    this.nodes.clear();
    this.liveNode = null;
    this.speakers.clear();
    this.segments = [];
    this.raw = '';
    $('transcript').innerHTML = '';
    $('empty-state').hidden = false;
    $('raw-text').textContent = '';
  },

  colorFor(speaker) {
    if (!speaker) return null;
    if (!this.speakers.has(speaker)) {
      this.speakers.set(speaker, SPEAKER_COLORS[this.speakers.size % SPEAKER_COLORS.length]);
    }
    return this.speakers.get(speaker);
  },

  initials(speaker) {
    if (!speaker) return '··';
    const digits = speaker.match(/\d+/);
    if (digits) return 'S' + digits[0];
    return speaker.replace(/[^A-Za-z一-鿿]/g, '').slice(0, 2).toUpperCase() || 'S';
  },

  ensure(index, speaker, chunkIndex) {
    let node = this.nodes.get(index);
    if (node) return node;

    $('empty-state').hidden = true;
    const color = this.colorFor(speaker);
    const wrap = document.createElement('div');
    wrap.className = 'utt is-live';
    wrap.dataset.speaker = speaker || 'none';
    wrap.dataset.start = ((chunkIndex || 0) * state.chunkSeconds).toFixed(2);

    const avatar = document.createElement('div');
    avatar.className = 'utt-avatar';
    avatar.textContent = this.initials(speaker);
    if (color) avatar.style.background = color;

    const body = document.createElement('div');
    body.className = 'utt-body';

    const name = document.createElement('div');
    name.className = 'utt-name';
    const nameText = document.createElement('span');
    nameText.textContent = speaker || 'Transcript';
    if (color) nameText.style.color = color;
    const time = document.createElement('span');
    time.className = 'utt-time';
    time.textContent = clock((chunkIndex || 0) * state.chunkSeconds);
    name.append(nameText, time);

    const text = document.createElement('div');
    text.className = 'utt-text';

    body.append(name, text);
    wrap.append(avatar, body);
    wrap.title = 'Click to seek the player here';
    wrap.addEventListener('click', () => {
      const player = $('player');
      if (player && player.src) { player.currentTime = parseFloat(wrap.dataset.start) || 0; player.play(); }
    });

    $('transcript').appendChild(wrap);
    node = { wrap, text, speaker };
    this.nodes.set(index, node);
    this.scroll();
    return node;
  },

  delta(event) {
    const node = this.ensure(event.segment, event.speaker, event.chunk);
    if (this.liveNode && this.liveNode !== node) this.liveNode.wrap.classList.remove('is-live');
    node.wrap.classList.add('is-live');
    this.liveNode = node;
    if (event.text) {
      node.text.textContent += event.text;
      this.scroll();
    }
  },

  // Reading scrollHeight forces layout, so coalesce to one check per frame --
  // deltas arrive per token, several dozen a second.
  scroll() {
    if (this._scrollQueued) return;
    this._scrollQueued = true;
    requestAnimationFrame(() => {
      this._scrollQueued = false;
      const box = $('transcript');
      if (box.scrollHeight - box.scrollTop - box.clientHeight < 160) box.scrollTop = box.scrollHeight;
    });
  },

  finish(payload) {
    if (this.liveNode) { this.liveNode.wrap.classList.remove('is-live'); this.liveNode = null; }
    if (payload) {
      this.segments = payload.segments || [];
      this.raw = payload.raw || '';
      $('raw-text').textContent = this.raw;
    }
  },

  appendRaw(chunk) {
    if (!chunk) return;
    this.raw += chunk;
    $('raw-text').textContent = this.raw;
  },

  plainText() {
    return Array.from(this.nodes.values())
      .map((n) => (n.speaker ? `${n.speaker}: ` : '') + n.text.textContent.trim())
      .filter((line) => line.replace(/^[^:]*:\s*$/, '').trim())
      .join('\n');
  },
};

/* -------------------------------------------------------------- telemetry */

const metrics = {
  reset() {
    state.latencies = [];
    ['m-chunks', 'm-tokens', 'm-audio', 'm-elapsed', 'm-rtf', 'm-lat-last', 'm-lat-avg', 'm-lat-first']
      .forEach((id) => { $(id).textContent = '—'; });
    $('rtf-fill').style.width = '0%';
    this.drawLatency();
  },

  update(stats) {
    if (!stats) return;
    $('m-chunks').textContent = stats.chunks ?? '—';
    $('m-tokens').textContent = stats.tokens ?? '—';
    $('m-audio').textContent = stats.audio_seconds != null ? clock(stats.audio_seconds) : '—';
    $('m-elapsed').textContent = stats.elapsed_seconds != null ? clock(stats.elapsed_seconds) : '—';

    if (stats.rtf != null) {
      $('m-rtf').textContent = stats.rtf.toFixed(2) + '×';
      const pct = Math.max(2, Math.min(100, (stats.rtf / 2) * 100));
      $('rtf-fill').style.width = pct + '%';
      $('m-rtf').style.color = stats.rtf <= 1 ? 'var(--ok)' : 'var(--warn)';
      $('rtf-note').textContent = stats.rtf <= 1
        ? `GPU time per second of audio — ${(1 / stats.rtf).toFixed(1)}× faster than real time.`
        : 'Slower than real time; live audio will queue up.';
    }
    if (stats.chunk_latency_ms_last != null) $('m-lat-last').textContent = Math.round(stats.chunk_latency_ms_last) + ' ms';
    if (stats.chunk_latency_ms_avg != null) $('m-lat-avg').textContent = Math.round(stats.chunk_latency_ms_avg) + ' ms';
    if (stats.first_token_ms_avg != null) $('m-lat-first').textContent = Math.round(stats.first_token_ms_avg) + ' ms';
  },

  pushLatency(ms) {
    state.latencies.push(ms);
    if (state.latencies.length > 64) state.latencies.shift();
    this.drawLatency();
  },

  drawLatency() {
    const canvas = $('latency-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 260;
    const h = 76;
    if (canvas.width !== Math.round(w * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const budget = state.chunkSeconds * 1000;
    const data = state.latencies;
    const peak = Math.max(budget * 1.15, ...data, 1);

    // real-time budget line
    const budgetY = h - (budget / peak) * (h - 10) - 5;
    ctx.strokeStyle = 'rgba(61,220,151,.45)';
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, budgetY); ctx.lineTo(w, budgetY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(61,220,151,.6)';
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText('real-time budget', 6, Math.max(9, budgetY - 4));

    if (!data.length) return;
    const gap = 2;
    const barW = Math.max(2, (w - gap * (data.length - 1)) / data.length);
    data.forEach((value, i) => {
      const barH = Math.max(2, (value / peak) * (h - 10));
      const x = i * (barW + gap);
      ctx.fillStyle = value <= budget ? '#3ddc97' : '#ffb454';
      ctx.globalAlpha = 0.35 + 0.65 * ((i + 1) / data.length);
      ctx.fillRect(x, h - barH - 5, barW, barH);
    });
    ctx.globalAlpha = 1;
  },
};

/* ------------------------------------------------------------ event sink */

function handleEvent(event) {
  switch (event.type) {
    case 'meta':
      if (event.frames) { state.chunkSeconds = event.frames.chunk_seconds; $('chunk-secs').textContent = event.frames.chunk_seconds.toFixed(1); }
      if (event.total_chunks) setStatus(`0 / ${event.total_chunks} chunks`, 'busy');
      break;
    case 'delta':
      transcript.delta(event);
      break;
    case 'chunk':
      metrics.update(event.stats);
      metrics.pushLatency(event.latency_ms);
      transcript.appendRaw(event.raw);
      setStatus(`chunk ${event.index + 1} · ${Math.round(event.latency_ms)} ms`, 'busy');
      break;
    case 'backlog':
      $('mic-backlog').textContent = event.seconds > 0.5 ? `queue ${event.seconds.toFixed(1)}s` : '';
      break;
    case 'notice':
      toast(event.message);
      break;
    case 'done':
      transcript.finish(event);
      metrics.update(event.stats);
      setStatus(`done · ${event.stats ? event.stats.chunks : 0} chunks`, 'done');
      break;
    case 'error':
      toast(event.message, true);
      setStatus('error', '');
      break;
  }
}

/* ----------------------------------------------------------------- health */

async function pollHealth() {
  let ready = false;
  try {
    const res = await fetch('/api/health');
    const info = await res.json();
    const dot = $('health-dot');
    const text = $('health-text');

    if (info.status === 'ready') {
      ready = true;
      dot.className = 'dot ok';
      text.textContent = 'model ready';
      $('loading-banner').hidden = true;
      if (!state.config) await loadConfig();
    } else if (info.status === 'loading') {
      dot.className = 'dot busy';
      text.textContent = 'loading model…';
      $('loading-banner').hidden = false;
      $('loading-detail').textContent = `17 GB of weights — ${Math.round(info.uptime_seconds)}s elapsed.`;
    } else {
      dot.className = 'dot err';
      text.textContent = 'backend error';
      $('loading-banner').hidden = false;
      $('loading-detail').textContent = info.error || 'see `docker compose logs backend`';
    }

    if (info.gpu) {
      $('gpu-pill').hidden = false;
      $('gpu-pill').textContent = `${info.gpu.name} · ${info.gpu.reserved_gb.toFixed(1)}/${info.gpu.total_gb.toFixed(0)} GB`;
    }
    renderModelInfo(info);
    setControlsEnabled(info.status === 'ready');
  } catch (err) {
    $('health-dot').className = 'dot err';
    $('health-text').textContent = 'backend unreachable';
    setControlsEnabled(false);
  }
  return ready;
}

function renderModelInfo(info) {
  const rows = [
    ['Checkpoint', 'VibeVoice-ASR-Streaming-7B'],
    ['Device', info.device],
    ['Precision', info.dtype],
    ['Attention', info.attn_implementation],
  ];
  if (info.frames) {
    rows.push(['Chunk', `${info.frames.chunk_seconds.toFixed(2)} s`]);
    rows.push(['Lookahead', `${info.frames.lookahead_seconds.toFixed(2)} s`]);
    rows.push(['Sample rate', `${info.frames.sample_rate / 1000} kHz`]);
  }
  if (info.load_seconds) rows.push(['Load time', `${info.load_seconds.toFixed(0)} s`]);
  if (state.config && state.config.languages) {
    rows.push(['Languages', state.config.languages.map((l) => l.code).join(' ')]);
  }
  $('model-kv').innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join('');
}

async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    if (!res.ok) return;
    state.config = await res.json();
    state.chunkSeconds = state.config.frames.chunk_seconds;
    $('chunk-secs').textContent = state.chunkSeconds.toFixed(1);
    $('lang-count').textContent = state.config.languages.length;
    metrics.drawLatency();
    await loadSamples();
  } catch (_) { /* retried on the next poll */ }
}

function setControlsEnabled(ready) {
  $('btn-record').disabled = !ready || state.busy;
  $('btn-transcribe').disabled = !ready || state.busy || !state.source;
  $('btn-compare').disabled = !ready || state.busy || !state.source;
}

/* --------------------------------------------------------------- samples */

async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    state.samples = await res.json();
  } catch (_) { state.samples = []; }
  renderSamples();
}

function sampleCard(sample, onPick) {
  const button = document.createElement('button');
  button.className = 'example';
  button.type = 'button';
  button.innerHTML =
    `<b>${sample.title}</b><span>${sample.blurb}</span>` +
    (sample.hotwords ? `<span class="tagline">hotwords: ${sample.hotwords}</span>` : '');
  button.addEventListener('click', () => onPick(sample, button));
  return button;
}

function renderSamples() {
  const list = $('examples-list');
  const compare = $('compare-source');
  list.innerHTML = '';
  compare.innerHTML = '';

  if (!state.samples.length) {
    list.innerHTML = '<p class="hint">No sample files mounted at <code>/samples</code>.</p>';
    compare.innerHTML = '<p class="hint">Pick a file in the File tab first.</p>';
    return;
  }

  for (const sample of state.samples) {
    list.appendChild(sampleCard(sample, async (s, node) => {
      list.querySelectorAll('.example').forEach((n) => n.classList.remove('is-active'));
      node.classList.add('is-active');
      await selectSample(s);
      runFileTranscription();
    }));
    compare.appendChild(sampleCard(sample, async (s, node) => {
      compare.querySelectorAll('.example').forEach((n) => n.classList.remove('is-active'));
      node.classList.add('is-active');
      await selectSample(s);
      setControlsEnabled(true);
    }));
  }
}

async function selectSample(sample) {
  setStatus('fetching sample…', 'busy');
  const res = await fetch(sample.url);
  const blob = await res.blob();
  setSource({ name: sample.file, blob, hotwords: sample.hotwords });
  if (sample.hotwords) $('hotwords').value = sample.hotwords;
  setStatus('idle');
}

function setSource({ name, blob, hotwords }) {
  if (state.source && state.source.url) URL.revokeObjectURL(state.source.url);
  const url = URL.createObjectURL(blob);
  state.source = { name, blob, url, hotwords: hotwords || '' };
  $('player').src = url;
  $('player-bar').hidden = false;
  $('player-label').textContent = `${name} · ${(blob.size / 1e6).toFixed(1)} MB`;
  $('drop-title').textContent = name;
  setControlsEnabled($('health-dot').className.includes('ok'));
}

/* ---------------------------------------------------------- file → SSE */

async function runFileTranscription() {
  if (!state.source || state.busy) return;
  const form = new FormData();
  form.append('file', state.source.blob, state.source.name);
  form.append('context_info', $('hotwords').value.trim());
  form.append('temperature', $('temperature').value);
  form.append('max_new_tokens', $('max-tokens').value);

  beginRun();
  $('btn-cancel-file').disabled = false;
  const controller = new AbortController();
  state.abort = controller;

  const player = $('player');
  if (player.src) { player.currentTime = 0; player.play().catch(() => {}); }

  try {
    const res = await fetch('/api/transcribe', { method: 'POST', body: form, signal: controller.signal });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    for await (const event of sseEvents(res, controller.signal)) handleEvent(event);
  } catch (err) {
    if (err.name !== 'AbortError') { toast(err.message, true); setStatus('error'); }
    else setStatus('cancelled');
  } finally {
    endRun();
    $('btn-cancel-file').disabled = true;
  }
}

/* ---------------------------------------------------------- A/B compare */

async function runCompare() {
  if (!state.source || state.busy) return;
  const hotwords = $('hotwords').value.trim();
  if (!hotwords) { toast('Enter some hotwords first.', true); return; }

  const form = new FormData();
  form.append('file', state.source.blob, state.source.name);
  form.append('context_info', hotwords);
  form.append('temperature', $('temperature').value);
  form.append('max_new_tokens', $('max-tokens').value);

  beginRun();
  $('compare-grid').hidden = false;
  document.querySelector('.transcript-wrap').hidden = true;
  $('compare-a').innerHTML = '';
  $('compare-b').innerHTML = '';
  $('compare-words').textContent = hotwords;

  const controller = new AbortController();
  state.abort = controller;
  const buffers = { without: '', with: '' };
  const lastSegment = { without: -1, with: -1 };
  const results = {};

  try {
    const res = await fetch('/api/compare', { method: 'POST', body: form, signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    for await (const event of sseEvents(res, controller.signal)) {
      if (event.type === 'run_start') {
        setStatus(event.run === 'without' ? 'run 1 of 2 — no hotwords' : 'run 2 of 2 — with hotwords', 'busy');
      } else if (event.type === 'delta' && event.run) {
        // Label turns while streaming so the finished text does not re-flow.
        if (event.segment !== lastSegment[event.run]) {
          lastSegment[event.run] = event.segment;
          if (event.speaker) buffers[event.run] += `${buffers[event.run] ? '\n' : ''}${event.speaker}: `;
        }
        buffers[event.run] += event.text || '';
        renderCompareColumn(event.run, buffers[event.run], hotwords);
      } else if (event.type === 'chunk') {
        metrics.update(event.stats);
        metrics.pushLatency(event.latency_ms);
      } else if (event.type === 'run_done') {
        results[event.run] = event;
        renderCompareColumn(event.run, event.text, hotwords);
      } else if (event.type === 'done') {
        setStatus('both runs done', 'done');
      } else if (event.type === 'error') {
        handleEvent(event);
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') toast(err.message, true);
  } finally {
    endRun();
  }
}

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function renderCompareColumn(run, text, hotwords) {
  const target = run === 'without' ? $('compare-a') : $('compare-b');
  let html = escapeHtml(text || '');
  const words = hotwords.split(/[,\n]/).map((w) => w.trim()).filter((w) => w.length > 1);
  for (const word of words) {
    const pattern = new RegExp(`(${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    html = html.replace(pattern, '<mark class="hit">$1</mark>');
  }
  target.innerHTML = `<div class="utt-text">${html}</div>`;
  target.scrollTop = target.scrollHeight;
}

/* -------------------------------------------------------------- live mic */

async function startRecording() {
  if (state.busy) return;
  const sampleRate = (state.config && state.config.frames.sample_rate) || 24000;

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    const why = {
      NotAllowedError: 'Microphone blocked — allow it for this site in Chrome and try again.',
      NotFoundError: 'No microphone found on this machine.',
      NotReadableError: 'The microphone is already in use by another application.',
      SecurityError: 'The browser needs a secure context — open the demo on http://localhost.',
    }[err.name];
    toast(why || `Microphone unavailable (${err.name}).`, true);
    return;
  }
  state.micStream = stream;

  try {
    await openLiveSession(stream, sampleRate);
  } catch (err) {
    // The mic is already hot at this point; never leave it running.
    stopRecording(false);
    if (err && err.message) toast(err.message, true);
  }
}

async function openLiveSession(stream, sampleRate) {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/ws/live`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  ws.onerror = () => toast('Live connection failed.', true);
  ws.onclose = () => { if (state.busy) stopRecording(false); };

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    setTimeout(() => reject(new Error('the backend did not accept the live connection')), 8000);
  });

  ws.send(JSON.stringify({
    context_info: $('hotwords').value.trim() || null,
    temperature: parseFloat($('temperature').value) || 0,
    max_new_tokens: parseInt($('max-tokens').value, 10) || 256,
  }));

  beginRun();
  $('btn-record').classList.add('is-recording');
  $('btn-record').disabled = true;
  $('btn-stop').disabled = false;
  setStatus('recording', 'live');

  const ctx = new AudioContext({ sampleRate });
  state.audioCtx = ctx;
  await ctx.audioWorklet.addModule('pcm-processor.js');
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'pcm-processor');
  state.workletNode = node;

  node.port.onmessage = (event) => {
    const pcm = new Float32Array(event.data);
    drawScope(pcm);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(pcm.buffer);
  };
  source.connect(node);
  // Keep the graph alive without echoing the mic back to the speakers.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  node.connect(mute).connect(ctx.destination);

  state.recordStart = performance.now();
  state.timerId = setInterval(() => {
    $('mic-timer').textContent = clock((performance.now() - state.recordStart) / 1000);
  }, 200);
}

function stopRecording(flush = true) {
  if (state.timerId) { clearInterval(state.timerId); state.timerId = null; }
  if (state.workletNode) { try { state.workletNode.port.onmessage = null; state.workletNode.disconnect(); } catch (_) {} state.workletNode = null; }
  if (state.micStream) { state.micStream.getTracks().forEach((t) => t.stop()); state.micStream = null; }
  if (state.audioCtx) { state.audioCtx.close().catch(() => {}); state.audioCtx = null; }

  $('btn-record').classList.remove('is-recording');
  $('btn-stop').disabled = true;

  if (flush && state.ws && state.ws.readyState === WebSocket.OPEN) {
    setStatus('flushing tail…', 'busy');
    state.ws.send('end');
    const socket = state.ws;
    socket.addEventListener('close', () => { state.ws = null; endRun(); }, { once: true });
    setTimeout(() => { if (state.ws === socket) { try { socket.close(); } catch (_) {} state.ws = null; endRun(); } }, 120000);
  } else {
    if (state.ws) { try { state.ws.close(); } catch (_) {} state.ws = null; }
    endRun();
  }
}

function drawScope(pcm) {
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
  const rms = Math.sqrt(sum / pcm.length);
  state.levels.push(Math.min(1, rms * 3.6));
  if (state.levels.length > 140) state.levels.shift();

  const canvas = $('scope');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 260;
  const h = 84;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const n = state.levels.length;
  const barW = Math.max(1.5, w / n - 1);
  const mid = h / 2;
  for (let i = 0; i < n; i++) {
    const level = state.levels[i];
    const barH = Math.max(2, level * (h - 12));
    const x = i * (w / n);
    const grad = ctx.createLinearGradient(0, mid - barH / 2, 0, mid + barH / 2);
    grad.addColorStop(0, '#6d8cff');
    grad.addColorStop(1, '#3ddc97');
    ctx.fillStyle = grad;
    ctx.globalAlpha = 0.25 + 0.75 * (i / n);
    ctx.fillRect(x, mid - barH / 2, barW, barH);
  }
  ctx.globalAlpha = 1;
}

/* ------------------------------------------------------------ run state */

function beginRun() {
  state.busy = true;
  transcript.reset();
  metrics.reset();
  $('compare-grid').hidden = true;
  document.querySelector('.transcript-wrap').hidden = false;
  setStatus('starting…', 'busy');
  setControlsEnabled(false);
  $('btn-stop').disabled = state.mode !== 'mic';
}

function endRun() {
  state.busy = false;
  state.abort = null;
  $('mic-backlog').textContent = '';
  $('btn-record').disabled = false;
  $('btn-stop').disabled = true;
  setControlsEnabled($('health-dot').className.includes('ok'));
}

/* ------------------------------------------------------------------ wire */

function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('is-active', tab.dataset.mode === mode));
  document.querySelectorAll('.mode-panel').forEach((panel) => { panel.hidden = panel.dataset.panel !== mode; });
  const inCompare = mode === 'compare' && !!$('compare-a').innerHTML;
  $('compare-grid').hidden = !inCompare;
  document.querySelector('.transcript-wrap').hidden = inCompare;
}

function init() {
  document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => switchMode(tab.dataset.mode)));

  $('btn-record').addEventListener('click', () => startRecording().catch((err) => {
    toast(err && err.message ? err.message : 'Could not start recording.', true);
    stopRecording(false);
  }));
  $('btn-stop').addEventListener('click', () => stopRecording(true));
  $('btn-transcribe').addEventListener('click', runFileTranscription);
  $('btn-compare').addEventListener('click', runCompare);
  $('btn-cancel-file').addEventListener('click', () => state.abort && state.abort.abort());

  $('file-input').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) setSource({ name: file.name, blob: file });
  });

  const zone = $('dropzone');
  ['dragenter', 'dragover'].forEach((type) => zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((type) => zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove('is-over'); }));
  zone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) { setSource({ name: file.name, blob: file }); switchMode('file'); }
  });

  $('temperature').addEventListener('input', (e) => { $('temp-val').textContent = parseFloat(e.target.value).toFixed(2); });
  $('max-tokens').addEventListener('input', (e) => { $('maxtok-val').textContent = e.target.value; });
  $('show-raw').addEventListener('change', (e) => { $('raw-panel').hidden = !e.target.checked; });

  $('btn-copy').addEventListener('click', async () => {
    const text = transcript.plainText();
    if (!text) { toast('Nothing to copy yet.'); return; }
    try {
      await navigator.clipboard.writeText(text);
      toast('Transcript copied.');
    } catch (_) {
      // navigator.clipboard needs a focused document; fall back to a selection.
      const scratch = document.createElement('textarea');
      scratch.value = text;
      scratch.setAttribute('readonly', '');
      scratch.style.cssText = 'position:fixed;top:-1000px;opacity:0';
      document.body.appendChild(scratch);
      scratch.select();
      const ok = document.execCommand('copy');
      scratch.remove();
      toast(ok ? 'Transcript copied.' : 'Clipboard blocked by the browser.', !ok);
    }
  });

  $('btn-json').addEventListener('click', () => {
    const payload = {
      model: 'microsoft/VibeVoice-ASR-Streaming-7B',
      source: state.source ? state.source.name : 'microphone',
      hotwords: $('hotwords').value.trim() || null,
      segments: Array.from(transcript.nodes.values()).map((n) => ({ speaker: n.speaker, text: n.text.textContent })),
      raw: transcript.raw,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'vibevoice-transcript.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });

  $('btn-clear').addEventListener('click', () => { transcript.reset(); metrics.reset(); setStatus('idle'); });

  window.addEventListener('resize', () => { metrics.drawLatency(); });

  metrics.reset();
  scheduleHealth(0);
}

// Poll hard while the checkpoint is loading, then back off -- a 4 s poll makes
// `docker compose logs -f backend` unreadable once everything is up.
function scheduleHealth(delay) {
  setTimeout(async () => {
    const ready = await pollHealth();
    scheduleHealth(ready ? 15000 : 3000);
  }, delay);
}

document.addEventListener('DOMContentLoaded', init);
