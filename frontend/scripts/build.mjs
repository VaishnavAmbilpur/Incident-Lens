/**
 * scripts/build.mjs
 * =================
 * Production build for IncidentLens UI.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * `vite build` is the normal, preferred path and it is tried FIRST. If it succeeds, this
 * script simply reports it - Vite's own output (code splitting, hashed assets, Vite's HTML
 * transform) is used untouched.
 *
 * On this machine the DSH file sandbox denies creating any child process that owns a pipe
 * (EACCES/EPERM on uv_pipe). esbuild's JavaScript API always communicates with its own
 * binary over a piped stdin, so it cannot start here and `vite build` dies with
 * "Error: spawn EPERM". That is an environment limit, not a project defect: on an
 * unrestricted machine the first path runs and this fallback is never used.
 *
 * The fallback therefore drives the real toolchain through process binaries whose stdio is
 * fully inherited (no pipe is ever created):
 *   - esbuild CLI  : JSX transform, "@/..." alias, bundling, minification, NODE_ENV define
 *   - tailwind CLI : src/index.css -> CSS with the Tailwind v3 + PostCSS pipeline applied
 * then assembles dist/index.html itself.
 *
 * Both paths emit the same shape: dist/index.html + dist/assets/*.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const DIST = join(ROOT, "dist");
const DIST_ASSETS = join(DIST, "assets");

/** Path to an installed package's CLI entry point. */
function pkgBin(pkg, entry) {
  return join(ROOT, "node_modules", pkg, entry);
}

/**
 * Resolve a package's executable entry by reading its `bin` field.
 * We run that file with `node` rather than the node_modules/.bin shim: a .cmd shim cannot be
 * spawned on Windows without a shell, and going through node keeps stdio fully inherited.
 */
function resolveBin(pkg) {
  const pkgJsonPath = join(ROOT, "node_modules", pkg, "package.json");
  const pkgJson = JSON.parse(readFileSync(pkgJsonPath, "utf8"));
  const bin = pkgJson.bin;
  const rel =
    typeof bin === "string"
      ? bin
      : bin && typeof bin === "object"
        ? bin[pkg] || Object.values(bin)[0]
        : null;
  if (!rel) throw new Error(`Cannot determine the bin entry for ${pkg}.`);
  return join(ROOT, "node_modules", pkg, rel);
}

function run(command, args) {
  // stdio is fully inherited: this is the whole point of the fallback.
  const result = spawnSync(command, args, { cwd: ROOT, stdio: "inherit" });
  if (result.error) throw result.error;
  return result.status ?? 1;
}

/** Can we create a child process with a pipe? If yes, Vite works normally. */
function pipesAreAvailable() {
  try {
    const probe = spawnSync(process.execPath, ["-e", "0"], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    return !probe.error;
  } catch {
    return false;
  }
}

function hasPackage(pkg) {
  return existsSync(join(ROOT, "node_modules", pkg, "package.json"));
}

/** Try `vite build`; returns true when Vite produced a dist/ we can trust. */
function runViteBuild() {
  if (!hasPackage("vite")) {
    console.warn("[build] vite is not installed; skipping the Vite path.");
    return false;
  }
  console.log("[build] Running `vite build` (primary path)...");
  const code = run(process.execPath, [pkgBin("vite", join("bin", "vite.js")), "build"]);
  if (code === 0 && existsSync(join(DIST, "index.html"))) return true;

  console.warn(
    `[build] "vite build" did not complete (exit ${code}); falling back to the CLI build.`
  );
  rmSync(DIST, { recursive: true, force: true });
  return false;
}

/**
 * Minimal .env reader so VITE_* variables work on the fallback path too.
 * Vite's own path handles .env natively; this keeps the two paths equivalent.
 * @returns {Record<string,string>}
 */
function readViteEnv(mode = "production") {
  const out = {};
  for (const file of [".env", `.env.${mode}`, ".env.local", `.env.${mode}.local`]) {
    const path = join(ROOT, file);
    if (!existsSync(path)) continue;
    for (const rawLine of readFileSync(path, "utf8").split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) continue;
      const eq = line.indexOf("=");
      if (eq === -1) continue;
      const key = line.slice(0, eq).trim();
      let value = line.slice(eq + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }
      if (key) out[key] = value;
    }
  }
  return out;
}

/** Fallback: esbuild CLI + tailwind CLI, then assemble index.html. */
function runCliBuild() {
  if (!hasPackage("esbuild") || !hasPackage("tailwindcss")) {
    console.error("[build] esbuild and/or tailwindcss are missing. Run `npm install` first.");
    return 1;
  }

  rmSync(DIST, { recursive: true, force: true });
  mkdirSync(DIST_ASSETS, { recursive: true });

  // Vite statically replaces `import.meta.env`; esbuild needs the same defines or the bundle
  // would reference an env object that does not exist in the browser. Both the whole object
  // and each key are defined: the object catches any access pattern, the keys keep the
  // replacement readable even if the object literal form is ever rejected.
  const viteEnv = readViteEnv("production");
  const envObject = {
    MODE: "production",
    DEV: false,
    PROD: true,
    BASE_URL: "/",
    ...viteEnv,
  };
  const envDefines = Object.entries(viteEnv).map(
    ([key, value]) => `--define:import.meta.env.${key}=${JSON.stringify(value)}`
  );

  console.log("[build] Bundling src/main.jsx with the esbuild CLI...");
  const esbuildCode = run(process.execPath, [
    resolveBin("esbuild"),
    "src/main.jsx",
    "--bundle",
    "--minify",
    "--format=esm",
    "--target=es2020",
    "--jsx=automatic",
    "--loader:.js=jsx",
    // The "@" alias from vite.config.js, expressed for esbuild.
    "--alias:@=./src",
    '--define:process.env.NODE_ENV="production"',
    `--define:import.meta.env=${JSON.stringify(envObject)}`,
    "--define:import.meta.env.DEV=false",
    "--define:import.meta.env.PROD=true",
    '--define:import.meta.env.MODE="production"',
    ...envDefines,
    "--outdir=dist/assets",
    "--entry-names=index",
    "--log-level=warning",
  ]);
  if (esbuildCode !== 0) {
    console.error("[build] esbuild failed.");
    return esbuildCode;
  }

  console.log("[build] Compiling Tailwind (v3) CSS...");
  const tailwindCode = run(process.execPath, [
    resolveBin("tailwindcss"),
    "-c",
    "tailwind.config.js",
    "-i",
    "src/index.css",
    "-o",
    "dist/assets/index.css",
    "--minify",
    "--log-level=warn",
  ]);
  if (tailwindCode !== 0) {
    console.error("[build] tailwind failed.");
    return tailwindCode;
  }

  // A path found by the esbuild shim is always dist/assets/index.js; assert it.
  const bundlePath = join(DIST_ASSETS, "index.js");
  if (!existsSync(bundlePath)) {
    console.error("[build] dist/assets/index.js was not produced.");
    return 1;
  }

  console.log("[build] Assembling dist/index.html...");
  const html = readFileSync(join(ROOT, "index.html"), "utf8");
  const assembled = html.replace(
    /<script type="module" src="\/src\/main\.jsx"><\/script>/,
    '<script type="module" crossorigin src="/assets/index.js"></script>'
  );
  if (assembled === html) {
    console.error('[build] Could not find the /src/main.jsx script tag in index.html.');
    return 1;
  }
  const withCss = assembled.replace(
    "</head>",
    '  <link rel="stylesheet" crossorigin href="/assets/index.css" />\n  </head>'
  );
  writeFileSync(join(DIST, "index.html"), withCss, "utf8");

  return 0;
}

const usePipes = pipesAreAvailable();
console.log(
  usePipes
    ? "[build] Child processes with pipes are available."
    : "[build] Sandbox detected: child processes with pipes are denied (esbuild's JS API cannot run)."
);

const ok = usePipes && runViteBuild();
const code = ok ? 0 : runCliBuild();

if (code === 0) {
  console.log("[build] Done -> dist/");
} else {
  console.error("[build] Build failed.");
}
process.exit(code);
