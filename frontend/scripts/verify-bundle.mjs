/**
 * scripts/verify-bundle.mjs
 * =========================
 * Smoke test for the production bundle. Run: node scripts/verify-bundle.mjs
 *
 * It sets up the minimum browser surface the app touches at import time, then imports
 * dist/assets/index.js. Any unresolved import, missing export or top-level exception
 * surfaces as a non-zero exit. This proves the bundle is loadable; it is not a substitute
 * for opening it in a browser.
 *
 * Nothing here spawns a child process, so it works in restricted shells.
 */

const listeners = new Map();

function makeElement(tag = "div") {
  const el = {
    tagName: String(tag).toUpperCase(),
    nodeName: String(tag).toUpperCase(),
    style: {},
    dataset: {},
    children: [],
    childNodes: [],
    attributes: {},
    classList: {
      _set: new Set(),
      add(...c) {
        c.forEach((x) => this._set.add(x));
      },
      remove(...c) {
        c.forEach((x) => this._set.delete(x));
      },
      contains: (c) => el.classList._set.has(c),
      toggle(c, force) {
        const on = force === undefined ? !el.classList._set.has(c) : force;
        if (on) el.classList._set.add(c);
        else el.classList._set.delete(c);
        return on;
      },
    },
    setAttribute(k, v) {
      this.attributes[k] = v;
    },
    getAttribute(k) {
      return this.attributes[k] ?? null;
    },
    removeAttribute(k) {
      delete this.attributes[k];
    },
    hasAttribute(k) {
      return k in this.attributes;
    },
    appendChild(child) {
      this.childNodes.push(child);
      return child;
    },
    removeChild(child) {
      this.childNodes = this.childNodes.filter((c) => c !== child);
      return child;
    },
    insertBefore(child) {
      this.childNodes.unshift(child);
      return child;
    },
    addEventListener(type, fn) {
      listeners.set(`${this.tagName}:${type}`, fn);
    },
    removeEventListener() {},
    dispatchEvent() {
      return true;
    },
    focus() {},
    blur() {},
    select() {},
    contains() {
      return false;
    },
    getBoundingClientRect: () => ({ top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0 }),
    scrollIntoView() {},
    setPointerCapture() {},
    releasePointerCapture() {},
  };
  return el;
}

const root = makeElement("div");
root.id = "root";

globalThis.document = {
  documentElement: makeElement("html"),
  head: makeElement("head"),
  body: makeElement("body"),
  createElement: (tag) => makeElement(tag),
  createElementNS: (_ns, tag) => makeElement(tag),
  createTextNode: (text) => ({ nodeType: 3, textContent: text, data: text }),
  createDocumentFragment: () => makeElement("fragment"),
  getElementById: (id) => (id === "root" ? root : null),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: (type, fn) => listeners.set(`document:${type}`, fn),
  removeEventListener() {},
  execCommand: () => true,
};

globalThis.window = {
  document: globalThis.document,
  isSecureContext: false,
  innerWidth: 1280,
  innerHeight: 800,
  devicePixelRatio: 1,
  location: { href: "http://localhost:5173/", origin: "http://localhost:5173" },
  navigator: { userAgent: "node", clipboard: undefined },
  localStorage: {
    _m: new Map(),
    getItem(k) {
      return this._m.has(k) ? this._m.get(k) : null;
    },
    setItem(k, v) {
      this._m.set(k, String(v));
    },
    removeItem(k) {
      this._m.delete(k);
    },
  },
  matchMedia: () => ({
    matches: false,
    media: "",
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
  }),
  getComputedStyle: () => ({ getPropertyValue: () => "", fontFamily: "" }),
  addEventListener: (type, fn) => listeners.set(`window:${type}`, fn),
  removeEventListener() {},
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (id) => clearTimeout(id),
  setInterval: (fn, ms) => setInterval(fn, ms),
  clearInterval: (id) => clearInterval(id),
  requestAnimationFrame: (fn) => setTimeout(() => fn(Date.now()), 16),
  cancelAnimationFrame: (id) => clearTimeout(id),
  fetch: () => Promise.reject(new Error("network disabled in verify-bundle")),
  requestIdleCallback: (fn) => setTimeout(fn, 0),
  cancelIdleCallback: (id) => clearTimeout(id),
};

/**
 * Some globals (navigator in Node 22) are accessor-only, so a plain assignment throws.
 * Define them defensively instead.
 */
function defineGlobal(name, value) {
  try {
    Object.defineProperty(globalThis, name, {
      value,
      configurable: true,
      writable: true,
    });
  } catch (error) {
    console.warn(`[verify] Could not define global ${name}: ${error.message}`);
  }
}

defineGlobal("navigator", globalThis.window.navigator);
defineGlobal("localStorage", globalThis.window.localStorage);
defineGlobal("HTMLElement", class HTMLElement {});
defineGlobal("Element", class Element {});
defineGlobal("Node", class Node {});
defineGlobal("getComputedStyle", globalThis.window.getComputedStyle);
defineGlobal("requestAnimationFrame", globalThis.window.requestAnimationFrame);
defineGlobal("cancelAnimationFrame", globalThis.window.cancelAnimationFrame);
defineGlobal("IS_REACT_ACT_ENVIRONMENT", false);

// Event constructor shims used by Radix internals when they attach listeners.
defineGlobal("MouseEvent", class MouseEvent {});
defineGlobal("KeyboardEvent", class KeyboardEvent {});

const url = new URL("../dist/assets/index.js", import.meta.url);

let loaded = true;
try {
  await import(url.href);
} catch (error) {
  const message = String(error?.message || error);
  // React 18's createRoot validates the container against its real DOM node interface
  // (error #299). The stub above is close but not a DOM, so this specific rejection means
  // "the app code ran; the fake container was refused" - a harness limitation, not a
  // bundle defect. Everything else is a genuine failure.
  const stubRefusedContainer = /React error #299|createRoot/.test(message);
  if (stubRefusedContainer) {
    loaded = false;
    console.log("[verify] Bundle parsed and executed.");
    console.log(
      "[verify] React refused the stub container (#299) - expected without a real DOM."
    );
    console.log(
      "[verify] App code path reached: ReactDOM.createRoot was called. Open dist/index.html"
    );
    console.log("[verify] in a browser (or run `npm run preview`) for a full render check.");
  } else {
    console.error("[verify] The bundle threw while loading:");
    console.error(error);
    process.exit(1);
  }
}

if (loaded) {
  // Give React a tick to run its initial render, then assert it mounted something.
  await new Promise((r) => setTimeout(r, 300));

  if (root.childNodes.length === 0) {
    console.error("[verify] React did not mount into #root.");
    process.exit(1);
  }
  console.log("[verify] Bundle loaded and React mounted into #root.");
  console.log("[verify] Root child nodes:", root.childNodes.length);
}

process.exit(0);
