/**
 * Theme helpers.
 *
 * The key and the default ("dark") are duplicated in the inline <script> in index.html,
 * which applies the class before first paint to avoid a light flash. Keep them in sync.
 */
export const THEME_STORAGE_KEY = "incidentlens-theme";
export const DEFAULT_THEME = "dark";

/** @returns {"dark" | "light"} */
export function getStoredTheme() {
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    return saved === "light" || saved === "dark" ? saved : DEFAULT_THEME;
  } catch {
    // Private mode / storage disabled: fall back to the default rather than throwing.
    return DEFAULT_THEME;
  }
}

export function applyTheme(theme) {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.style.colorScheme = theme;
}

export function persistTheme(theme) {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* non-fatal: theme simply will not persist */
  }
}
