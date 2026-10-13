# IncidentLens UI

A modern, dark-first web interface for the **IncidentLens** production RAG API
(`modules/20_api_server.py`). This is a **frontend-only** project: the Python backend is
untouched and is called directly from the browser over CORS.

Built with **Vite + React 18 + plain JavaScript (.jsx)** and **Tailwind CSS v3** using
**shadcn/ui** conventions (Radix primitives, `class-variance-authority`, CSS-variable theme,
`components.json`).

---

## Requirements

| Tool | Version used |
| --- | --- |
| Node.js | 22.14.0 (18+ works) |
| npm | 11.5.2 |

## Install

```powershell
cd il_ui\frontend
npm install
```

`.npmrc` sets `ignore-scripts=true`. Nothing here needs a lifecycle script — esbuild's platform
binary arrives through its own optional dependency and the Tailwind CLI is a plain Node entry
point — and skipping them is what lets the install succeed in restricted shells.

> If `npm install` fails with `EPERM` while writing the **global** npm cache (common in a
> sandboxed shell), point the cache at a folder you own and retry:
>
> ```powershell
> $env:npm_config_cache = "$PWD\..\.npm-cache"
> npm install
> ```

## Run the backend (required - the UI is useless without it)

The API is started from the **backend project root**, not from this folder:

```powershell
cd "C:\Users\Vaishnav Ambilpur\Desktop\Applied Ai\Production-Level-RAG"
.\venv\Scripts\python.exe modules\20_api_server.py
```

It listens on `http://127.0.0.1:8000` by default. The UI also shows this exact command in
its **API offline** state.

## Run the UI

```powershell
npm run dev      # http://localhost:5173
npm run build    # production bundle into dist/
npm run preview  # serve the built bundle
```

Open the dev URL, then confirm the status pill in the header reads **API online**.

### About `npm run build`

`npm run build` runs `scripts/build.mjs`, which **tries `vite build` first** and uses Vite's
output if it succeeds. If Vite cannot start, it falls back to driving the same toolchain
through the `esbuild` and `tailwindcss` command-line binaries and assembles `dist/index.html`
itself. Both paths emit `dist/index.html` + `dist/assets/*`.

The fallback exists for one reason: this project was built inside the **DSH file sandbox**,
which denies creating any child process that owns a pipe. esbuild's JavaScript API speaks to
its own binary over a piped stdin, so `vite build` and `vite dev` both die with
`Error: spawn EPERM` in that shell. The CLI binaries only need inherited stdio, so they work.
See **Restricted shells** below for what that means for `npm run dev`.

`npm run build:vite` is the direct Vite build, useful on an unrestricted machine.

### Verifying the build

```powershell
node scripts/verify-api-parsing.mjs   # SSE framing + Prometheus parsing (10 checks)
node scripts/verify-bundle.mjs        # loads dist/assets/index.js and reports if it runs
```

`verify-api-parsing.mjs` runs against byte-for-byte samples of what module 20 emits.
`verify-bundle.mjs` proves the bundle parses and executes; with no DOM available it cannot
prove a full React render - open `dist/index.html` in a browser for that.

### Restricted shells (DSH sandbox and similar)

If child processes with pipes are denied:

| Command | Result | Why |
| --- | --- | --- |
| `npm install` | works as-is | `.npmrc` sets `ignore-scripts=true`; only the global cache path may need redirecting |
| `npm run build` | works | `scripts/build.mjs` falls back to the CLI binaries |
| `npm run dev` | **fails with `spawn EPERM`** | Vite loads its own config through esbuild, which needs the piped service |
| `npm run preview` | fails for the same reason | same config loading step |

There is no Vite setting that avoids this: every JSX file goes through esbuild's service.
Run `npm run dev` from a normal (unsandboxed) shell - the project itself is unmodified and
standard. The production build in `dist/` can be served by any static file server in the
meantime, for example:

```powershell
npx serve dist     # or: python -m http.server 5173 --directory dist
```


## Pointing at a different API

The base URL is read once, at build time:

```js
// src/lib/api.js
export const API_BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");
```

Create a `.env` file next to `package.json`:

```dotenv
VITE_API_BASE=http://127.0.0.1:8000
```

`VITE_*` variables are inlined by Vite, so **restart the dev server** (or rebuild) after
changing it. CORS is already open on the server (`INCIDENTLENS_CORS_ORIGINS`, default `*`).

---

## What the app does

One page, four tabs, no router.

### Ask
- Large question textarea; **Ctrl/Cmd + Enter** submits.
- Optional **conversation id** — auto-filled from the last response and reused for
  follow-ups, which is what preserves the agent's thread memory.
- **Stream progress** switch:
  - **on** → `POST /query/stream`; the **agent trace** renders node by node as SSE frames
    arrive, with the active node pulsing.
  - **off** → `POST /query`; you get the full scorecard including relevance and grounding.
- The answer card carries a copy button, citations as monospace badges, latency, relevance
  (score **and** source), grounding (score or **NOT CHECKED**), and the full node path.
- A blocked query renders a red-accent card with the security note and states plainly that a
  blocked query costs **zero LLM calls**.

**Honest limitation, surfaced in the UI:** `/query/stream` only returns `answer`, `blocked`,
`citations` and `latency_ms`. Relevance and grounding are only produced by `POST /query`, so
in stream mode the answer card says so instead of showing fabricated numbers.

### Evaluate
Limit selector (1–5) with the 5-LLM-calls-per-question warning, skeletons while running, the
four metrics as cards comparing value against `targets` (with a target marker on each bar),
a judge-failures/complete banner, and the `per_question` table.

### Metrics
`GET /metrics` is parsed from Prometheus text into a key/value map and rendered as stat cards
(requests, queries, blocked, failed, latency avg/max, uptime) plus the raw exposition in a
scroll area.

### Health
The dependency checks as a list with status dots and `detail` text, service version and
timestamp, and a re-check button. The header pill polls `/health` every 15 s and is clickable.

### UI states implemented
| State | Where |
| --- | --- |
| Loading | Skeletons in the trace, the metric cards and the per-question table |
| Empty | Trace, answer, scorecard and metrics each have a written empty state |
| Offline | Red header pill + a banner carrying the exact start command; actions disabled |
| Degraded | Amber pill, amber summary on the Health tab |
| Blocked | Red-accent card + toast, "zero LLM calls" stated |
| Unverified | Amber badges: `external / unverified`, `relevance unknown`, `grounding NOT CHECKED` |
| Incomplete scorecard | Red `INCOMPLETE scorecard` banner whenever `judge_failures > 0` |

---

## Project layout

```
il_ui/frontend/
├── components.json            shadcn config ("tsx": false)
├── index.html                 includes the pre-paint theme script
├── package.json
├── postcss.config.js          classic Tailwind v3 pipeline
├── tailwind.config.js         classic config + CSS-variable theme
├── vite.config.js             "@" -> ./src alias
├── README.md
├── scripts/
│   ├── build.mjs              vite build first, CLI fallback (see above)
│   ├── verify-api-parsing.mjs parser unit checks
│   └── verify-bundle.mjs      loads the built bundle
└── src/
    ├── main.jsx
    ├── App.jsx                shell + tabs + all run state
    ├── index.css              light AND dark design tokens
    ├── lib/
    │   ├── api.js             all five calls, SSE reader, Prometheus parser
    │   ├── theme.js           localStorage theme helpers
    │   └── utils.js           cn()
    ├── hooks/
    │   ├── useHealth.js       polls /health, never throws into React
    │   └── useCopyToClipboard.js
    └── components/
        ├── header.jsx  status-pill.jsx  theme-toggle.jsx
        ├── query-panel.jsx  agent-trace.jsx  answer-card.jsx
        ├── evaluation-panel.jsx  metrics-panel.jsx  health-panel.jsx
        ├── stat-tile.jsx
        └── ui/               button, card, input, textarea, badge, tabs,
                              separator, scroll-area, skeleton, tooltip,
                              alert, label, switch, sonner
```

## Extending with the shadcn CLI

`components.json` is configured for JavaScript, so the CLI writes `.jsx` files:

```powershell
npx shadcn@latest add dialog
```

The CLI is **not** required to build or run this project — every component it would generate
is already written by hand into `src/components/ui/`.

## Notes

- No API keys, no secrets, no backend calls other than the five documented endpoints.
- No TypeScript: no `typescript` dependency, no `.ts`/`.tsx` files.
- The backend project folder is never written to.
