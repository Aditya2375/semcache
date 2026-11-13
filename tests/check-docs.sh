#!/bin/sh
# check-docs.sh - the README's claims must match the code and data. When a
# pinned number changes on purpose, change it here and in README.md together.
set -eu
cd "$(dirname "$0")/.."
fail() { echo "check-docs: $1" >&2; exit 1; }

UNIT=$(npm test 2>&1 | grep -o '^# pass [0-9]*' | grep -o '[0-9]*')
[ "$UNIT" = "58" ] || fail "unit+integration tests: README says 58, runner says $UNIT"
grep -q "58 unit + integration tests" README.md || fail "README no longer says 58 unit + integration tests"
grep -q "19 end-to-end checks" README.md || fail "README lost the smoke check count"
grep -q "aditya2375.github.io/semcache" README.md || fail "README lost the live demo link"

# replay table pins (regenerate with node scripts/replay.mjs)
for row in "fixed tau=0.75|199|24.6% \\[19.2-31.1\\]|27.2% \\[21.2-34.1\\]|53.6%|71.0%" \
           "fixed tau=0.85|134|73.9% \\[65.9-80.6\\]|55.0% \\[47.7-62.1\\]|12.5%|47.7%" \
           "fixed tau=0.92|93|97.8% \\[92.5-99.4\\]|50.6% \\[43.3-57.8\\]|0.7%|33.1%" \
           "margin(tau=0.75, m=0.05)|150|79.3% \\[72.2-85.0\\]|66.1% \\[58.9-72.6\\]|11.1%|53.2%" \
           "calibrated(p\\*=0.95)|60|100.0% \\[94.0-100.0\\]|33.3% \\[26.9-40.5\\]|0.0%|21.0%"; do
  pat=$(echo "$row" | sed 's/|/ *| */g')
  grep -qE "^\\| $pat" README.md || fail "README replay row missing: $row"
done
grep -q 'AUC 0.819' README.md || fail "README lost the dev AUC"
grep -q 'log-loss 0.542' README.md || fail "README lost the dev log-loss"
grep -q 'Brier 0.176' README.md || fail "README lost the dev Brier"
grep -q '41.1%' README.md || fail "README lost the p*=0.90 recall"
grep -q 'seed 20261113' README.md || fail "README lost the data seed"

# model pin agrees between the browser adapter, the precompute script and README
PIN="ea104dacec62c0de699686887e3f920caeb4f3e3"
grep -q "$PIN" src/web/models.ts || fail "models.ts lost the revision pin"
grep -q "$PIN" scripts/precompute-core.mjs || fail "precompute-core.mjs lost the revision pin"
grep -q 'ea104dac' README.md || fail "README lost the embed model revision"
grep -q "$PIN" data/precomputed/helpdesk.bge-small-en-v1.5.json || fail "artifact lost the revision pin"

# data pins
QCOUNT=$(node -e "console.log(JSON.parse(require('fs').readFileSync('data/helpdesk/workload.json','utf8')).queries.length)")
[ "$QCOUNT" = "280" ] || fail "workload query count changed: $QCOUNT"
CCOUNT=$(node -e "console.log(JSON.parse(require('fs').readFileSync('data/helpdesk/canonical.json','utf8')).canonicals.length)")
[ "$CCOUNT" = "60" ] || fail "canonical count changed: $CCOUNT"
TRAPS=$(node -e "const w=JSON.parse(require('fs').readFileSync('data/helpdesk/workload.json','utf8'));console.log(w.queries.filter(q=>q.forbidCanonicalIds).length)")
[ "$TRAPS" = "24" ] || fail "near-miss trap count changed: $TRAPS"
grep -q "280-query" README.md || fail "README lost the workload size"

# built site is present and self-contained
for f in docs/index.html docs/app.js docs/worker.js docs/style.css \
         docs/ort/ort-wasm-simd-threaded.mjs docs/ort/ort-wasm-simd-threaded.wasm \
         docs/data/helpdesk/workload.json docs/data/helpdesk/canonical.json \
         docs/data/helpdesk/calibration.json docs/data/helpdesk/pairs.json \
         docs/data/precomputed/helpdesk.bge-small-en-v1.5.json; do
  [ -s "$f" ] || fail "$f missing - run node scripts/build-web.mjs"
done
grep -q "Content-Security-Policy" docs/index.html || fail "docs/index.html lost its CSP"
grep -q "wasm-unsafe-eval" docs/index.html || fail "CSP lost the wasm permission ONNX needs"

# no stray tooling mentions anywhere public-facing. Authored text only:
# minified bundles, base64 artifacts and wasm can contain look-alike bytes.
! grep -rli --exclude=check-docs.sh "claude\|anthropic\|chatgpt\|copilot\|impeccable\|cursor\.com\|vibe.cod\|windsurf" \
  README.md SECURITY_REVIEW.md src/ scripts/ tests/ data/helpdesk data/precomputed \
  .github/ package.json docs/index.html docs/style.css \
  || fail "tooling mention found"
echo "check-docs ok"
