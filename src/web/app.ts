/**
 * UI controller: tabs, the query console, workload replay with a live event
 * feed, the threshold sweep with canvas curves, the near-miss gallery and
 * the method tab. All rendering goes through textContent - no HTML ever
 * flows from data into the DOM.
 */

/* eslint-disable @typescript-eslint/no-non-null-assertion */

type PolicySpec =
  | { kind: 'fixed'; tau: number }
  | { kind: 'margin'; tau: number; margin: number }
  | { kind: 'calibrated'; pStar?: number; wrongMultiple?: number };

interface RpcResponse {
  id: number;
  ok: boolean;
  result?: unknown;
  error?: string;
  progress?: number;
}

class WorkerClient {
  private nextId = 1;
  private readonly pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (f: number) => void }>();
  private readonly worker = new Worker('./worker.js');

  constructor() {
    this.worker.onmessage = (ev: MessageEvent<RpcResponse>) => {
      const msg = ev.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (typeof msg.progress === 'number') {
        p.onProgress?.(msg.progress);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.ok) p.resolve(msg.result);
      else p.reject(new Error(msg.error ?? 'worker error'));
    };
  }

  call<T>(kind: string, args: Record<string, unknown> = {}, onProgress?: (f: number) => void): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, ...(onProgress ? { onProgress } : {}) });
      this.worker.postMessage({ id, kind, ...args });
    });
  }
}

const client = new WorkerClient();

function el<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`missing #${id}`);
  return e as T;
}

function div(className: string, text?: string): HTMLDivElement {
  const d = document.createElement('div');
  d.className = className;
  if (text !== undefined) d.textContent = text;
  return d;
}

const fmtUsd = (x: number) => `$${x.toFixed(6)}`;
const fmtPct = (x: number) => `${(x * 100).toFixed(1)}%`;
const fmtPct100 = (x: number) => `${x.toFixed(1)}%`;

/* ---------------- tabs ---------------- */
const tabs = [...document.querySelectorAll<HTMLButtonElement>('.tab')];
for (const tab of tabs) {
  tab.addEventListener('click', () => {
    for (const t of tabs) {
      const on = t === tab;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', String(on));
      const panel = document.getElementById(`tab-${t.dataset['tab']!}`);
      if (panel) (panel as HTMLElement).hidden = !on;
    }
  });
}

/* ---------------- policy pickers ---------------- */
function policyFromSelect(select: HTMLSelectElement): PolicySpec {
  switch (select.value) {
    case 'fixed':
      return { kind: 'fixed', tau: 0.85 };
    case 'margin':
      return { kind: 'margin', tau: 0.75, margin: 0.05 };
    default:
      return { kind: 'calibrated', wrongMultiple: 10 };
  }
}

/* ---------------- console ---------------- */
const queryForm = el<HTMLFormElement>('query-form');
const queryInput = el<HTMLTextAreaElement>('query-input');
const querySubmit = el<HTMLButtonElement>('query-submit');
const decisionCard = el<HTMLDivElement>('decision-card');
const spendCard = el<HTMLDivElement>('spend-card');
const modelStatus = el<HTMLDivElement>('model-status');

for (const chip of document.querySelectorAll<HTMLButtonElement>('.chip')) {
  chip.addEventListener('click', () => {
    queryInput.value = chip.dataset['q'] ?? '';
    queryInput.focus();
  });
}

interface QueryResult {
  kind: string;
  reason?: string;
  answer: string;
  matchedQuery?: string;
  similarity?: number;
  p?: number | null;
  bestSimilarity?: number | null;
  entryHits?: number;
  latencyMs: number;
  costUsd: number;
  trace: Array<{ stage: string; detail: string }>;
  spend: { lookupUsd: number; generationUsd: number; baselineUsd: number };
  modelsUsed: string;
}

queryForm.addEventListener('submit', (ev) => {
  ev.preventDefault();
  const text = queryInput.value.trim();
  if (text.length === 0) return;
  querySubmit.disabled = true;
  const select = el<HTMLSelectElement>('policy-select');
  modelStatus.hidden = false;
  modelStatus.textContent = 'embedding model: checking bundle';
  client
    .call<QueryResult>('query', { text, policy: policyFromSelect(select) }, (f) => {
      modelStatus.hidden = false;
      modelStatus.textContent = `downloading embedding model: ${(f * 100).toFixed(0)}%`;
    })
    .then((r) => renderDecision(r))
    .catch((err: Error) => renderError(err))
    .finally(() => {
      querySubmit.disabled = false;
      modelStatus.hidden = true;
    });
});

function renderError(err: Error): void {
  decisionCard.classList.remove('is-empty');
  decisionCard.replaceChildren();
  const v = div('verdict');
  const k = div('verdict-kind guard', 'error');
  v.append(k);
  decisionCard.append(v, div('answer', err.message));
}

function renderDecision(r: QueryResult): void {
  decisionCard.classList.remove('is-empty');
  decisionCard.replaceChildren();

  const verdict = div('verdict');
  const kindText =
    r.kind === 'hit-exact' ? 'HIT · exact' : r.kind === 'hit-semantic' ? 'HIT · semantic' : r.kind === 'guard' ? 'GUARDED' : 'MISS';
  const kindClass = r.kind.startsWith('hit') ? 'hit' : r.kind === 'guard' ? 'guard' : 'miss';
  verdict.append(div(`verdict-kind ${kindClass}`, kindText));
  const meta: string[] = [`${r.latencyMs.toLocaleString()} ms`, fmtUsd(r.costUsd)];
  if (r.similarity !== undefined && r.kind === 'hit-semantic') meta.push(`sim ${r.similarity!.toFixed(3)}`);
  if (r.p !== undefined && r.p !== null) meta.push(`p(correct) ${r.p.toFixed(3)}`);
  verdict.append(div('verdict-meta mono', meta.join('  ·  ')));
  decisionCard.append(verdict);

  const trace = document.createElement('ol');
  trace.className = 'trace';
  for (const t of r.trace) {
    const li = document.createElement('li');
    li.append(div('t-stage', t.stage), div('t-detail', t.detail));
    trace.append(li);
  }
  decisionCard.append(trace);

  decisionCard.append(div('answer', r.answer));
  const src: string[] = [`model path: ${r.modelsUsed}`];
  if (r.matchedQuery !== undefined) src.push(`served from: "${r.matchedQuery}"`);
  if (r.reason !== undefined && r.kind !== 'guard') src.push(r.reason);
  if (r.kind === 'guard' && r.reason !== undefined) src.push(r.reason);
  decisionCard.append(div('answer-src', src.join('  ·  ')));

  spendCard.hidden = false;
  el<HTMLSpanElement>('spend-cache').textContent = fmtUsd(r.spend.lookupUsd + r.spend.generationUsd);
  el<HTMLSpanElement>('spend-baseline').textContent = fmtUsd(r.spend.baselineUsd);
  el<HTMLSpanElement>('spend-saved').textContent = fmtUsd(r.spend.baselineUsd - r.spend.lookupUsd - r.spend.generationUsd);
}

/* ---------------- workload replay ---------------- */
interface ReplayEventRow {
  queryId: string;
  text: string;
  category: string;
  actual: string;
  similarity: number | null;
  p: number | null;
}
interface ReplaySummary {
  total: number;
  hits: number;
  correctHits: number;
  wrongHits: number;
  missedHits: number;
  correctMisses: number;
  guardOk: number;
  guardMissed: number;
  hitRecall: number;
  hitRecallCi: [number, number];
  hitPrecision: number;
  hitPrecisionCi: [number, number];
  wrongHitRate: number;
  wrongHitRateCi: [number, number];
  guardAccuracy: number;
  costCacheUsd: number;
  costBaselineUsd: number;
  costSavedUsd: number;
  costSavedPct: number;
  p50LatencyCacheMs: number;
  p95LatencyCacheMs: number;
  p50LatencyBaselineMs: number;
  p95LatencyBaselineMs: number;
}
interface ReplayResult {
  events: ReplayEventRow[];
  summary: ReplaySummary;
}

el<HTMLButtonElement>('replay-run').addEventListener('click', () => {
  const btn = el<HTMLButtonElement>('replay-run');
  btn.disabled = true;
  const select = el<HTMLSelectElement>('replay-policy-select');
  client
    .call<ReplayResult>('replay', { policy: policyFromSelect(select) })
    .then((r) => renderReplay(r))
    .catch((err: Error) => renderError(err))
    .finally(() => (btn.disabled = false));
});

function stat(value: string, key: string, cls = '', ci = ''): HTMLDivElement {
  const s = div('stat');
  s.append(div(`stat-v ${cls}`, value), div('stat-k', key));
  if (ci !== '') s.append(div('stat-ci', ci));
  return s;
}

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function renderReplay(r: ReplayResult): void {
  const s = r.summary;
  const box = el<HTMLDivElement>('replay-summary');
  box.hidden = false;
  box.replaceChildren();
  const grid = div('summary-grid');
  grid.append(
    stat(`${s.hits}`, 'cache hits', '', `of ${s.total} queries`),
    stat(fmtPct(s.hitPrecision), 'hit precision', s.hitPrecision >= 0.99 ? 'ok' : s.hitPrecision < 0.8 ? 'bad' : 'accent', `95% CI ${fmtPct(s.hitPrecisionCi[0])}-${fmtPct(s.hitPrecisionCi[1])}`),
    stat(fmtPct(s.hitRecall), 'hit recall', '', `95% CI ${fmtPct(s.hitRecallCi[0])}-${fmtPct(s.hitRecallCi[1])}`),
    stat(fmtPct(s.wrongHitRate), 'wrong-hit rate', s.wrongHitRate === 0 ? 'ok' : 'bad', `95% CI ${fmtPct(s.wrongHitRateCi[0])}-${fmtPct(s.wrongHitRateCi[1])}`),
    stat(`${s.guardOk}/${s.guardOk + s.guardMissed}`, 'guards held', s.guardMissed === 0 ? 'ok' : 'bad'),
    stat(fmtPct100(s.costSavedPct), 'cost saved', 'accent', `${fmtUsd(s.costCacheUsd)} vs ${fmtUsd(s.costBaselineUsd)}`),
    stat(`${s.p50LatencyCacheMs.toFixed(0)} ms`, 'p50 latency', '', `baseline ${s.p50LatencyBaselineMs.toFixed(0)} ms`),
    stat(`${s.p95LatencyCacheMs.toFixed(0)} ms`, 'p95 latency', '', `baseline ${s.p95LatencyBaselineMs.toFixed(0)} ms`),
  );
  box.append(grid);

  const feed = el<HTMLOListElement>('event-feed');
  const counters = el<HTMLDivElement>('feed-counters');
  counters.hidden = false;
  feed.replaceChildren();
  const counts = { 'correct-hit': 0, 'wrong-hit': 0, 'missed-hit': 0, 'correct-miss': 0, 'guard-ok': 0, 'guard-missed': 0 } as Record<string, number>;
  const updateCounters = () => {
    counters.textContent = `${counts['correct-hit']} correct hits · ${counts['wrong-hit']} wrong hits · ${counts['missed-hit']} missed · ${counts['correct-miss']} correct misses · ${counts['guard-ok']} guards`;
  };
  updateCounters();

  const step = reducedMotion ? r.events.length : 1;
  let i = 0;
  const tick = () => {
    for (let n = 0; n < step && i < r.events.length; n++, i++) {
      const e = r.events[i]!;
      counts[e.category] = (counts[e.category] ?? 0) + 1;
      const li = document.createElement('li');
      li.append(div('ev-id', e.queryId), div(`ev-cat ${e.category}`, e.category), div('ev-text', e.text));
      feed.append(li);
    }
    updateCounters();
    if (i < r.events.length) {
      feed.scrollTop = feed.scrollHeight;
      window.setTimeout(tick, reducedMotion ? 0 : 45);
    }
  };
  tick();
}

/* ---------------- threshold sweep ---------------- */
interface CurvePoint {
  param: number;
  precision: number;
  recall: number;
  wrongHitRate: number;
  costSavedPct: number;
}
interface SweepResult {
  fixedCurve: CurvePoint[];
  calibratedCurve: CurvePoint[];
  devMetrics: { auc: number; logloss: number; brier: number; pairs: number };
}

el<HTMLButtonElement>('sweep-run').addEventListener('click', () => {
  const btn = el<HTMLButtonElement>('sweep-run');
  btn.disabled = true;
  btn.textContent = 'replaying 23 workloads…';
  client
    .call<SweepResult>('sweep')
    .then((r) => renderSweep(r))
    .catch((err: Error) => renderError(err))
    .finally(() => {
      btn.disabled = false;
      btn.textContent = 'Compute the curves';
    });
});

function drawAxes(ctx: CanvasRenderingContext2D, w: number, h: number, xLabel: string, yLabel: string): void {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#262d36';
  ctx.fillStyle = '#6b7480';
  ctx.lineWidth = 1;
  ctx.font = '11px ui-monospace, monospace';
  const padL = 44;
  const padB = 34;
  const padT = 26;
  const padR = 14;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, h - padB);
  ctx.lineTo(w - padR, h - padB);
  ctx.stroke();
  for (let i = 0; i <= 4; i++) {
    const y = padT + ((h - padB - padT) * i) / 4;
    ctx.fillText(`${(100 - i * 25).toFixed(0)}%`, 8, y + 4);
    ctx.strokeStyle = '#1b2129';
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
  }
  ctx.fillText(xLabel, w / 2 - ctx.measureText(xLabel).width / 2, h - 8);
  ctx.fillText(yLabel, 8, padT - 14);
}

function plotCurve(
  ctx: CanvasRenderingContext2D,
  pts: Array<{ x: number; y: number }>,
  color: string,
  w: number,
  h: number,
): void {
  const padL = 44;
  const padB = 34;
  const padT = 26;
  const padR = 14;
  const X = (v: number) => padL + v * (w - padL - padR);
  const Y = (v: number) => padT + (1 - v) * (h - padT - padB);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((p, i) => (i === 0 ? ctx.moveTo(X(p.x), Y(p.y)) : ctx.lineTo(X(p.x), Y(p.y))));
  ctx.stroke();
  for (const p of pts) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(X(p.x), Y(p.y), 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderSweep(r: SweepResult): void {
  const pr = el<HTMLCanvasElement>('pr-canvas');
  const wh = el<HTMLCanvasElement>('wh-canvas');
  const prCtx = pr.getContext('2d')!;
  const whCtx = wh.getContext('2d')!;
  drawAxes(prCtx, pr.width, pr.height, 'recall →', 'precision');
  drawAxes(whCtx, wh.width, wh.height, 'threshold (tau fixed · p* calibrated) →', 'wrong-hit rate');

  const fixedPR = r.fixedCurve.map((p) => ({ x: p.recall, y: p.precision }));
  const calPR = r.calibratedCurve.map((p) => ({ x: p.recall, y: p.precision }));
  plotCurve(prCtx, fixedPR, '#9aa4b0', pr.width, pr.height);
  plotCurve(prCtx, calPR, '#e3a008', pr.width, pr.height);

  const normFixed = (v: number) => (v - 0.7) / 0.28;
  const normCal = (v: number) => (v - 0.5) / 0.49;
  plotCurve(whCtx, r.fixedCurve.map((p) => ({ x: normFixed(p.param), y: p.wrongHitRate })), '#9aa4b0', wh.width, wh.height);
  plotCurve(whCtx, r.calibratedCurve.map((p) => ({ x: normCal(p.param), y: p.wrongHitRate })), '#e3a008', wh.width, wh.height);

  const tableBox = el<HTMLDivElement>('sweep-table');
  tableBox.hidden = false;
  tableBox.replaceChildren();
  const legend = div('panel-note');
  legend.textContent = `Grey: fixed threshold sweep (tau 0.70-0.98). Amber: calibrated policy sweep (p* 0.50-0.99). Calibration dev-fold AUC ${r.devMetrics.auc.toFixed(3)} on ${r.devMetrics.pairs} held-out intent pairs (logloss ${r.devMetrics.logloss.toFixed(3)}, brier ${r.devMetrics.brier.toFixed(3)}).`;
  tableBox.append(legend);

  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  for (const h of ['policy', 'precision', 'recall', 'wrong-hit', 'cost saved']) {
    const th = document.createElement('th');
    th.textContent = h;
    hr.append(th);
  }
  thead.append(hr);
  table.append(thead);
  const tbody = document.createElement('tbody');
  const rows: Array<[string, CurvePoint]> = [
    ...r.fixedCurve.filter((p) => [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98].includes(p.param)).map((p): [string, CurvePoint] => [`fixed tau=${p.param.toFixed(2)}`, p]),
    ...r.calibratedCurve.map((p): [string, CurvePoint] => [`calibrated p*=${p.param.toFixed(2)}`, p]),
  ];
  for (const [label, p] of rows) {
    const tr = document.createElement('tr');
    if (label === 'fixed tau=0.92' || label === 'calibrated p*=0.90') tr.className = 'hl';
    for (const cell of [label, fmtPct(p.precision), fmtPct(p.recall), fmtPct(p.wrongHitRate), fmtPct100(p.costSavedPct)]) {
      const td = document.createElement('td');
      td.className = 'mono';
      td.textContent = cell;
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(tbody);
  tableBox.append(table);
}

/* ---------------- gallery ---------------- */
interface GalleryRow {
  queryId: string;
  text: string;
  note: string;
  naiveOutcome: string;
  calibratedOutcome: string;
}

el<HTMLButtonElement>('gallery-run').addEventListener('click', () => {
  const btn = el<HTMLButtonElement>('gallery-run');
  btn.disabled = true;
  client
    .call<{ rows: GalleryRow[] }>('gallery')
    .then((r) => {
      const box = el<HTMLDivElement>('gallery-table');
      box.replaceChildren();
      for (const row of r.rows) {
        const card = div('g-row');
        card.append(div('g-q', row.text));
        if (row.note !== '') card.append(div('g-note', row.note));
        const cols = div('g-outcomes');
        const naiveBad = row.naiveOutcome.startsWith('wrong hit');
        const calBad = !row.calibratedOutcome.startsWith('miss (correct)');
        const left = div(`g-col ${naiveBad ? 'bad' : 'good'}`);
        const lt = document.createElement('span');
        lt.className = 'g-col-title';
        lt.textContent = 'fixed tau = 0.85';
        left.append(lt, document.createTextNode(row.naiveOutcome));
        const right = div(`g-col ${calBad ? 'bad' : 'good'}`);
        const rt = document.createElement('span');
        rt.className = 'g-col-title';
        rt.textContent = 'calibrated (cost-derived p*)';
        right.append(rt, document.createTextNode(row.calibratedOutcome));
        cols.append(left, right);
        card.append(cols);
        box.append(card);
      }
    })
    .catch((err: Error) => renderError(err))
    .finally(() => (btn.disabled = false));
});

/* ---------------- method ---------------- */
function renderMethod(dev: { auc: number; logloss: number; brier: number; pairs: number }): void {
  const body = el<HTMLDivElement>('method-body');
  body.replaceChildren();
  const add = (tag: string, text: string) => {
    const e = document.createElement(tag);
    e.textContent = text;
    body.append(e);
  };
  add('h3', 'The decision, precisely');
  add('p', 'A lookup walks four stages: cacheability guards, an exact normalized-key check, approximate nearest neighbours over query embeddings (HNSW, cosine), then the hit policy. The calibrated policy estimates P(the nearest entry answers the same question) from four features - top similarity, similarity margin over the runner-up, cache size, query length - with a logistic model fitted on 120 labeled intent pairs, split by intent group so paraphrases never leak across folds.');
  const f = document.createElement('div');
  f.className = 'formula';
  f.textContent = 'serve cached  <=>  (1 - p) * W  <=  C      =>      p >= 1 - C/W';
  body.append(f);
  add('p', 'W is what a wrong cached answer costs you; C is what regenerating costs. At W/C = 10 the operating point is p* = 0.90. The ratio is a product decision made explicit, not a hidden constant.');
  add('h3', 'Calibration quality (held out)');
  add('p', `On ${dev.pairs} dev pairs the fit reaches AUC ${dev.auc.toFixed(3)}, log-loss ${dev.logloss.toFixed(3)}, brier ${dev.brier.toFixed(3)}. Honest reading: useful, not magic - the margin feature contributed little on this workload's hard negatives, and the model knows less than a production judge model would.`);
  add('h3', 'What the numbers mean');
  const ul = document.createElement('ul');
  for (const li of [
    'Every rate carries a 95% Wilson interval. A rate without its interval is a marketing number, not a measurement.',
    'The backend is simulated: answers come from bundled NimbusDB documentation, latency and token counts from a seeded generator. Dollar figures use published gpt-4o-mini prices (0.15/0.60 USD per 1M input/output tokens, openai.com/api/pricing, retrieved 2026-11-13) and are illustrative.',
    'Embeddings: bge-small-en-v1.5 at pinned revision ea104dac, quantized, running in wasm. Workload vectors are precomputed and verified in CI (cosine >= 0.995 on regeneration); custom console queries embed live with the same model.',
    'The workload, labels and near-miss traps are hand-authored for a fictional product; no user data appears anywhere.',
  ]) {
    const e = document.createElement('li');
    e.textContent = li;
    ul.append(e);
  }
  body.append(ul);
  add('h3', 'Known limits');
  const ul2 = document.createElement('ul');
  for (const li of [
    'Guards are lexical: they catch time-relative, private-state and imperative phrasings they know about, and miss phrasings they do not. Production would layer a classifier; the lexical floor stays for explainability.',
    'Calibration is fitted per embedding model and per domain. Swap either and refit - the artifact pins its model revision for exactly this reason.',
    'TTL and LRU are the only eviction disciplines. Event-driven invalidation (doc updated -> drop its answers) is real work this demo does not do.',
    'A semantic cache can never raise answer quality above the backend\'s; it can only return a good answer faster, or a bad one instantly.',
  ]) {
    const e = document.createElement('li');
    e.textContent = li;
    ul2.append(e);
  }
  body.append(ul2);
}

/* ---------------- boot ---------------- */
interface BootStats {
  canonicals: number;
  workload: number;
  texts: number;
  embedder: string;
  devMetrics: { auc: number; logloss: number; brier: number; pairs: number };
  price: { model: string; asOf: string; source: string };
}

client
  .call<BootStats>('init')
  .then((b) => {
    el<HTMLParagraphElement>('boot-stats').textContent =
      `${b.workload} labeled queries · ${b.canonicals} documentation entries · ${b.texts} bundled vectors · ${b.embedder} · prices: ${b.price.model} as of ${b.price.asOf} (illustrative)`;
    renderMethod(b.devMetrics);
  })
  .catch((err: Error) => {
    el<HTMLParagraphElement>('boot-stats').textContent = `failed to initialize: ${err.message}`;
  });
