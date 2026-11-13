# Security review - semcache

Item-by-item pass over the pre-ship checklist, with evidence. Statuses: PASS
(checked, evidence below), N/A (the thing the item protects does not exist in
this project - stated explicitly, not skipped silently), TODO (must happen
before or right after publication).

Architecture context for every item below: semcache is a fully static site.
No backend, no database, no auth, no accounts, no cookies, no analytics, no
API keys, no forms, no uploads. The only network calls the page makes are
same-origin (its own files) and HTTPS model downloads from the pinned
Hugging Face hosts named in the CSP. The console's cache lives only in the
Web Worker's memory and vanishes with the tab.

## Pre-ship routine

1. **Read the diff** - PASS. All source (src/, scripts/, tests/, docs/) was
   reviewed line by line; there is no generated or copied code outside
   docs/ort/ (vendored onnxruntime-web binaries) and the bundled
   app.js/worker.js, which are built from src/ and diff-checked in CI.
2. **gitleaks detect** - PASS. CI job `supply-chain` runs gitleaks 8.24.3
   (sha256-verified binary) over the full git history on every push. The
   full-history scan initially flagged 2 hits, both one false positive: the
   tokenizer special-token name "unk_token" inside the vendored, minified
   ONNX Runtime bundle docs/worker.js (line 13). Allowlisted in
   .gitleaks.toml (vendored artifacts only); scan then exits 0 locally.
   No keys exist anywhere in the project by design (no backends to key),
   and a manual scan of src/, scripts/, tests/, README.md, data/ found no
   key/token/secret strings.
3. **Dependency audit** - PASS. `npm audit`: 0 vulnerabilities. Runtime
   dependency is @huggingface/transformers (pinned 4.3.2) with its
   onnxruntime-web dependency; esbuild/typescript are dev-only. CI reruns
   npm audit.
4. **Static scan** - PASS. CI job `supply-chain` runs semgrep with the
   repo's own local rules (.github/semgrep-rules.yml, no account) over src/
   and scripts/ on every push. Local evidence: manual review; the
   risky-pattern grep is part of tests/check-docs.sh.
5. **Logged-out test** - N/A. There is nothing to log in to. Every byte the
   site serves is public by design; the console cache is per-tab memory.
6. **MDN Observatory scan** - N/A, verified by hand instead. Observatory
   scans hosts, not paths: scanning aditya2375.github.io returns "Site
   responded with an unexpected HTTP status code 404" because there is no
   user-level Pages site at the apex (2026-11-13). Verified directly
   instead: `curl -sI https://aditya2375.github.io/semcache/` returns 200
   with strict-transport-security: max-age=31556952 (GitHub Pages default
   for the *.github.io domain); Pages cannot send the remaining security
   headers, so CSP is delivered via <meta> (see item 17) and the meta tag
   is confirmed present in the served live HTML. Accepted per the
   checklist's own Pages guidance.
7. **Push public only after checks** - PASS (process). Nothing was pushed
   before this review, check-docs, unit tests, browser smoke, and the
   precomputed-verify gate all passed locally.

## Secrets and keys

8. **(1) No secret in code, config, bundle or history** - PASS. The project
   has no secrets to leak. CI runs gitleaks over full git history. GitHub
   secret scanning + push protection to be enabled on the repo at creation
   (free on public repos).
9. **(2) Keys meant to be public** - N/A. No keys of any kind ship. DevTools
   search of the live bundle finds no key material (only the model revision
   hash, a public identifier).
10. **(3) Gitignore secrets before first commit** - PASS. .gitignore covers
    .env*, *.keystore, *.jks, local.properties, *service-account*.json and
    the local model cache (.cache/) since the first commit.
11. **(4) Rotate anything that ever leaked** - N/A. Nothing ever existed to
    leak.

## Access control and data

12. **(5) RLS on every table** - N/A. No database.
13. **(6) Storage buckets private by default** - N/A. No buckets; nothing
    persists outside the tab.
14. **(7) Authorization on the server** - N/A. No server, no per-user
    resources.
15. **(8) Collect less** - PASS. The site collects nothing: no analytics, no
    telemetry, no cookies. The workload is hand-written text about a
    fictional product with no personal data; demo content contains no names,
    emails, phones, or addresses (verified by grep).

## Injection and input handling

16. **(9) No untrusted text into innerHTML/eval/DOM HTML** - PASS.
    `grep -rn "innerHTML\|eval(\|new Function" src/` matches only comments
    stating the rule. All query, answer, trace, and table text renders
    through textContent/createElement. The browser smoke test includes a
    live XSS probe (`<img src=x onerror=...>` as a console query) and asserts
    it cannot execute. CSP has no unsafe-inline and no unsafe-eval (only
    'wasm-unsafe-eval' for ONNX).
17. **(10) Parameterized queries; no shell strings** - N/A. No SQL, no shell
    execution. All JSON artifacts are parsed with JSON.parse and validated
    field-by-field (src/core/workload.ts, src/core/precomputed.ts).
18. **(11) Validate URLs and file paths** - PASS. The page fetches only five
    fixed relative data paths plus Hugging Face model URLs allowed by CSP
    connect-src; no user input is ever turned into a fetch URL. The console
    cache is capped at 2,000 entries; the precomputed artifact is validated
    (version, model identity, dims, per-row length) before use.

## Dependencies and supply chain

19. **(12) Every package exists and is the right one** - PASS. Dependencies
    are @huggingface/transformers, onnxruntime-web (transitive), esbuild,
    typescript - all long-established, verified on npmjs.com before install.
    package-lock.json pins exact versions and integrity hashes.
20. **(13) Commit lockfiles and audit** - PASS. package-lock.json is
    committed; npm audit is clean and runs in CI.
21. **(14) Treat AI config files as code** - PASS. The repo contains no
    AI-assistant instruction files of any kind. No code or README content
    was copied from external repos; everything here is written for this
    project.
22. **(15) Agents never get production access** - N/A. There is no
    production data or environment; the site is static and its data is
    public content.

## Config, transport and accounts

23. **(16) HTTPS only** - PASS. Pages serves the site over HTTPS with
    HSTS (curl-verified live, item 6); "Enforce HTTPS" is on and
    non-disableable for the default aditya2375.github.io domain ("Required
    for your site because you are using the default domain" - Pages
    settings, 2026-11-13). The site itself makes no http:// requests
    (grep-verified).
24. **(17) Security headers** - PASS within Pages' limits. CSP via <meta>:
    default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src
    'self'; connect-src 'self' plus the pinned Hugging Face download hosts;
    img-src 'self' data:; worker-src 'self'; object-src 'none'; base-uri
    'none'; form-action 'none'. frame-ancestors cannot be set via meta
    (documented Pages limitation).
25. **(18) Debug off, errors quiet, source maps off** - PASS. esbuild runs
    with minify and no sourcemaps; no .map files ship in docs/. Errors
    surface as short messages, not stack traces; nothing logs sensitive
    state (there is none).
26. **(19) Login provider, never homemade auth** - N/A. No auth anywhere.
27. **(20) Rate limits and abuse controls** - PASS (client-side). The
    console cache caps at 2,000 entries with LRU eviction; the sweep runs a
    bounded 23 replays; all compute is same-machine and bounded. No public
    endpoint exists to abuse.
28. **(21) Fail closed and log the failure** - PASS. Artifact validation
    rejects on the first violated rule; texts outside the precomputed
    artifact are never silently accepted as precomputed (the embedder throws
    and the console falls back to the live model explicitly). Unknown worker
    messages get an explicit error. Model download failure leaves the
    bundled demo fully functional.
29. **(22) Two-factor on accounts** - N/A to this repo's code; account-level
    setting on GitHub is the owner's own checklist item and stays with them.

## Static sites, GitHub Pages and Vercel

30. **(23) Everything in the published folder is public** - PASS. docs/
    contains only the built site, vendored wasm, and the public
    workload/calibration/precomputed data. Searched for phone, email,
    address, and personal names: none present. Git author identity is the
    owner's public GitHub noreply address only.
31. **(24) No client-side login or hidden admin page** - N/A. No login, no
    admin, no hidden routes.
32. **(25) Vercel secrets** - N/A. Hosted on GitHub Pages, not Vercel; and
    no secrets exist regardless.
33. **(26) Pin or self-host third-party scripts and fonts** - PASS. Zero CDN
    scripts, zero web fonts, zero external assets: app.js, worker.js and the
    ONNX wasm are self-hosted in docs/; model weights download only from the
    pinned Hugging Face hosts at an exact content revision. system-ui font
    stack only.
34. **(27) Forms go to a service with spam protection** - N/A. No forms.
35. **(28) Delete old deployments** - N/A to this new repo; owner's standing
    account hygiene item.

## Android apps (items 29-36, 37-39)

36. **(29-39) APK, manifest, keystore, WebView, relay channel, update
    hashes** - N/A. This project ships no Android app and no update channel.

## Python tools and the pension monitor (items 40-45)

37. **(40-43) Credentials, read-only monitors, log scrubbing, safe
    parsing** - N/A. No Python, no monitor, no credentials. (Item 44 applies
    to this repo's CI and is answered below.)
38. **(44) GitHub Actions least privilege** - PASS. The CI workflow sets
    `permissions: contents: read` at the top level, uses no secrets, and
    pins every third-party action to a full commit SHA.
39. **(45) Alerts to a private channel** - N/A. The project sends no alerts.

## Residual risks accepted on purpose

- **Model weight supply chain**: weights download from Hugging Face at a
  pinned revision; a compromised upstream revision would still be sandboxed
  by the CSP (no exfil host is whitelisted beyond the HF hosts themselves).
- **The lexical guard floor is incomplete by design**: phrasings it does not
  know reach the embedding stage. Documented in the Method tab; the guard is
  a floor, not a classifier.
- **CI minutes**: the workflow runs on push/PR only; no scheduled runs.
