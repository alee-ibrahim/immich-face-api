// Docs page: language tabs (one choice for all examples, remembered) and copy buttons. Served from the app itself,
// so the strict Content-Security-Policy (no inline script) holds.
(function () {
  const KEY = "faceapi-docs-lang";
  const groups = Array.from(document.querySelectorAll("[data-tabs]"));

  function choose(lang) {
    groups.forEach((g) => {
      const has = g.querySelector(`.pane[data-lang="${lang}"]`);
      const pick = has ? lang : g.querySelector(".pane").dataset.lang;
      g.querySelectorAll(".tab").forEach((t) => {
        const on = t.dataset.lang === pick;
        t.classList.toggle("on", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      });
      g.querySelectorAll(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.lang !== pick));
    });
    try { localStorage.setItem(KEY, lang); } catch (e) { /* private mode */ }
  }

  groups.forEach((g) => g.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => choose(t.dataset.lang))));
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* ignore */ }
  if (groups.length) choose(saved || "curl");

  document.querySelectorAll("button.copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const code = btn.parentElement.querySelector("code");
      try {
        await navigator.clipboard.writeText(code.innerText);
        btn.textContent = "Copied";
      } catch (e) {
        const range = document.createRange();
        range.selectNodeContents(code);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = "Selected";
      }
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    });
  });
})();
