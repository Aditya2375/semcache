/**
 * Browser smoke test: serves docs/, drives headless Chrome over CDP, and
 * checks the demo actually works - boot, console miss/hit/semantic-hit,
 * guard, replay summary, sweep curves, gallery, XSS safety. Runs locally
 * and in CI (ubuntu-latest ships google-chrome). No npm deps.
 */
import { spawn } from 'node:child_process';

const HTTP_PORT = 8932;
const CDP_PORT = 9334;
const root = new URL('..', import.meta.url).pathname;

const server = spawn('python3', ['-m', 'http.server', String(HTTP_PORT), '--bind', '127.0.0.1', '--directory', `${root}docs`], { stdio: 'ignore' });
const chrome = spawn('google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  `--remote-debugging-port=${CDP_PORT}`, 'about:blank',
], { stdio: 'ignore' });

const cleanup = (code) => {
  try { chrome.kill('SIGKILL'); } catch {}
  try { server.kill('SIGKILL'); } catch {}
  process.exit(code);
};
setTimeout(() => { console.error('smoke: timeout'); cleanup(1); }, 180000);
process.on('SIGINT', () => cleanup(2));

async function waitFor(fn, label, timeoutMs = 60000) {
  const start = Date.now();
  for (;;) {
    try {
      const v = await fn();
      if (v) return v;
    } catch {}
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${label}`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

const checks = [];
const check = (name, ok, detail = '') => {
  checks.push({ name, ok, detail });
  console.log(`${ok ? 'ok' : 'NOT OK'} - ${name}${detail ? ' - ' + detail : ''}`);
};

await waitFor(async () => (await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`)).ok, 'chrome devtools');
const target = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/new`, { method: 'PUT' })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });

let mid = 0;
const rpcPending = new Map();
const consoleErrors = [];
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && rpcPending.has(msg.id)) {
    rpcPending.get(msg.id)(msg);
    rpcPending.delete(msg.id);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    consoleErrors.push(msg.params.exceptionDetails?.exception?.description ?? 'page exception');
  }
};
const cdp = (method, params = {}) => new Promise((resolve) => {
  const id = ++mid;
  rpcPending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const res = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (res.result?.exceptionDetails) throw new Error(JSON.stringify(res.result.exceptionDetails));
  return res.result?.result?.value;
};

await cdp('Runtime.enable');
await cdp('Page.enable');
await cdp('Page.navigate', { url: `http://127.0.0.1:${HTTP_PORT}/` });

// Boot: worker loads artifacts and fills the boot stats line.
await waitFor(() => evaluate(`(document.getElementById('boot-stats').textContent || '').includes('280 labeled queries')`), 'boot');
check('boot loads workload, canonicals, vectors and calibration', true);

const ask = async (q) => {
  await evaluate(`(() => {
    const i = document.getElementById('query-input');
    i.value = ${JSON.stringify(q)};
    document.getElementById('query-form').dispatchEvent(new Event('submit', { cancelable: true }));
  })()`);
  await waitFor(() => evaluate(`!document.getElementById('query-submit').disabled && !document.getElementById('decision-card').classList.contains('is-empty')`), `answer for ${q.slice(0, 30)}`);
  return evaluate(`({
    verdict: document.querySelector('#decision-card .verdict-kind')?.textContent ?? '',
    meta: document.querySelector('#decision-card .verdict-meta')?.textContent ?? '',
    answer: document.querySelector('#decision-card .answer')?.textContent ?? '',
    traceStages: [...document.querySelectorAll('#decision-card .trace .t-stage')].map((e) => e.textContent),
  })`);
};

const Q1 = 'How do I migrate from MongoDB to NimbusDB?';
const Q2 = 'How do I move from MongoDB to NimbusDB?';

const first = await ask(Q1);
check('first ask is a MISS (cold cache)', first.verdict === 'MISS', first.verdict);
check('miss walks guard, exact, decision, store stages', ['guard','exact','decision','store'].every((st) => first.traceStages.includes(st)), first.traceStages.join(','));

const second = await ask(Q1);
check('repeat of the same question is an exact HIT', second.verdict === 'HIT · exact', second.verdict);

const third = await ask(Q2);
check('paraphrase is a semantic HIT', third.verdict === 'HIT · semantic', `${third.verdict} ${third.meta}`);
check('semantic hit reports similarity and p', /sim 0\.9\d\d/.test(third.meta) && /p\(correct\) 0\./.test(third.meta), third.meta);
check('hit names the cached source question', (await evaluate(`document.querySelector('#decision-card .answer-src').textContent`)).includes(Q1));

const guard = await ask('Show me my usage for this month');
check('private-state query is GUARDED, not cached', guard.verdict === 'GUARDED', guard.verdict);

const spend = await evaluate(`({
  card: !document.getElementById('spend-card').hidden,
  cache: document.getElementById('spend-cache').textContent,
  baseline: document.getElementById('spend-baseline').textContent,
  saved: document.getElementById('spend-saved').textContent,
})`);
check('spend counters track cache vs baseline', spend.card && /^\$0\.\d{6}$/.test(spend.cache) && /^\$0\.\d{6}$/.test(spend.baseline), JSON.stringify(spend));

// Replay tab (calibrated default).
await evaluate(`(() => { document.querySelector('.tab[data-tab="workload"]').click(); document.getElementById('replay-run').click(); })()`);
await waitFor(() => evaluate(`!document.getElementById('replay-summary').hidden && document.querySelectorAll('#replay-summary .stat').length === 8`), 'replay summary', 120000);
const replayStats = await evaluate(`[...document.querySelectorAll('#replay-summary .stat')].map((s) => s.querySelector('.stat-k').textContent + '=' + s.querySelector('.stat-v').textContent)`);
check('replay summary shows 8 measured stats', replayStats.length === 8, replayStats.join(' | '));
check('replay reports wrong-hit rate and cost saved', replayStats.some((s) => s.startsWith('wrong-hit rate=')) && replayStats.some((s) => s.startsWith('cost saved=')));
await waitFor(() => evaluate(`document.querySelectorAll('#event-feed li').length === 280`), 'event feed', 120000);
check('event feed replays all 280 labeled queries', true);
const counters = await evaluate(`document.getElementById('feed-counters').textContent`);
check('feed counters reconcile to 280', /^\d+ correct hits · \d+ wrong hits · \d+ missed · \d+ correct misses · \d+ guards$/.test(counters)
  && counters.split('·').map((s) => parseInt(s)).reduce((a, b) => a + b, 0) === 280, counters);

// Sweep tab.
await evaluate(`(() => { document.querySelector('.tab[data-tab="thresholds"]').click(); document.getElementById('sweep-run').click(); })()`);
await waitFor(() => evaluate(`!document.getElementById('sweep-table').hidden && document.querySelectorAll('#sweep-table tbody tr').length > 10`), 'sweep', 120000);
const sweepRows = await evaluate(`document.querySelectorAll('#sweep-table tbody tr').length`);
check('sweep table lists fixed and calibrated operating points', sweepRows > 10, String(sweepRows));
const prPixels = await evaluate(`(() => {
  const c = document.getElementById('pr-canvas');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let n = 0;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
  return n;
})()`);
check('precision-recall canvas is drawn', prPixels > 800, `${prPixels} px`);

// Gallery tab.
await evaluate(`(() => { document.querySelector('.tab[data-tab="gallery"]').click(); document.getElementById('gallery-run').click(); })()`);
await waitFor(() => evaluate(`document.querySelectorAll('#gallery-table .g-row').length > 10`), 'gallery', 120000);
const gallery = await evaluate(`({
  rows: document.querySelectorAll('#gallery-table .g-row').length,
  cols: document.querySelectorAll('#gallery-table .g-row')[0]?.querySelectorAll('.g-col').length ?? 0,
  titles: [...document.querySelectorAll('#gallery-table .g-col-title')].slice(0, 2).map((e) => e.textContent),
})`);
check('near-miss gallery renders paired outcomes', gallery.rows > 10 && gallery.cols === 2 && gallery.titles[0].includes('0.85'), JSON.stringify(gallery));

// Method tab.
const method = await evaluate(`(() => {
  document.querySelector('.tab[data-tab="method"]').click();
  return document.getElementById('method-body').textContent;
})()`);
check('method tab states the serving rule and honest limits', method.includes('serve cached') && method.includes('Known limits') && method.includes('simulated'));

// Live model: a custom query absent from the precomputed artifact must
// download the pinned model and embed it locally (CSP must allow the wasm).
await evaluate(`(() => {
  document.querySelector('.tab[data-tab="console"]').click();
  const i = document.getElementById('query-input');
  i.value = 'What does NimbusDB cost per month for a small team?';
  document.getElementById('query-form').dispatchEvent(new Event('submit', { cancelable: true }));
})()`);
await waitFor(() => evaluate(`!document.getElementById('query-submit').disabled`), 'live-model answer', 150000);
const live = await evaluate(`({
  verdict: document.querySelector('#decision-card .verdict-kind')?.textContent ?? '',
  src: document.querySelector('#decision-card .answer-src')?.textContent ?? '',
  status: document.getElementById('model-status')?.textContent ?? '',
})`);
check('custom query embeds with the live pinned model', live.src.includes('Xenova/bge-small-en-v1.5@ea104dac'), JSON.stringify(live).slice(0, 160));

// XSS: query text must never become markup (guard path needs no live model).
await evaluate(`(() => {
  document.querySelector('.tab[data-tab="console"]').click();
  const i = document.getElementById('query-input');
  i.value = '<img src=x onerror="window.__xss=1"> show me my usage this month';
  document.getElementById('query-form').dispatchEvent(new Event('submit', { cancelable: true }));
})()`);
await waitFor(() => evaluate(`!document.getElementById('query-submit').disabled`), 'xss answer');
const xss = await evaluate(`window.__xss === 1 || document.querySelector('#decision-card img') !== null`);
check('query text cannot inject markup', !xss);

check('no page exceptions during the whole run', consoleErrors.length === 0, consoleErrors[0] ?? '');

const failed = checks.filter((c) => !c.ok);
console.log(failed.length === 0 ? `smoke ok: ${checks.length} checks` : `smoke FAILED: ${failed.length}/${checks.length}`);
cleanup(failed.length === 0 ? 0 : 1);
