// Forms with data-confirm ask before submitting (inline onsubmit handlers are blocked by the CSP).
document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (ev) => {
    if (!window.confirm(form.dataset.confirm)) ev.preventDefault();
  });
});
