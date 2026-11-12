// "Try it": the photo is shown from the browser's own memory (blob URL) and sent once to /try; the answer holds
// only boxes and names. Nothing is stored on the server.
(function () {
  const form = document.getElementById("try-form");
  if (!form) return;
  const input = form.querySelector("input[type=file]");
  const button = form.querySelector("button");
  const stage = document.getElementById("stage");
  const img = document.getElementById("photo");
  const out = document.getElementById("result");
  const msg = document.getElementById("message");
  const csrf = form.dataset.csrf;
  let url = null;

  function text(el, value) { el.textContent = value; }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const file = input.files[0];
    if (!file) return;
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(file);
    stage.querySelectorAll(".face").forEach((el) => el.remove());
    out.replaceChildren();
    text(msg, "Detecting faces…");
    msg.className = "notice";
    button.disabled = true;
    const body = new FormData();
    body.append("image", file);
    try {
      const res = await fetch("/try", { method: "POST", body, headers: { "X-CSRF-Token": csrf }, credentials: "same-origin" });
      const data = await res.json();
      if (!res.ok) throw new Error((data.error && data.error.message) || res.statusText);
      img.src = url;
      stage.classList.remove("hidden");
      const known = data.faces.filter((f) => f.name).length;
      text(msg, `${data.faces.length} face(s), ${known} named. Image ${data.image.w}×${data.image.h}.`);
      data.faces.forEach((f, i) => {
        const box = document.createElement("div");
        box.className = "face" + (f.name ? " known" : "");
        box.style.left = (f.box.x * 100) + "%";
        box.style.top = (f.box.y * 100) + "%";
        box.style.width = (f.box.w * 100) + "%";
        box.style.height = (f.box.h * 100) + "%";
        const label = document.createElement("span");
        text(label, `${i + 1}. ${f.name || "unknown"}`);
        box.appendChild(label);
        stage.appendChild(box);
        const tr = document.createElement("tr");
        const cells = [
          String(i + 1),
          f.name || "unknown",
          f.distance == null ? "—" : f.distance.toFixed(3),
          f.ambiguous ? "yes" : "",
          (f.candidates || []).map((c) => `${c.name || "?"} (${c.distance.toFixed(3)})`).join(", "),
        ];
        const heads = ["#", "Name", "Distance", "Ambiguous", "Candidates"];
        cells.forEach((v, j) => { const td = document.createElement("td"); td.dataset.label = heads[j]; text(td, v); tr.appendChild(td); });
        out.appendChild(tr);
      });
    } catch (err) {
      text(msg, "Failed: " + err.message);
      msg.className = "notice error";
    } finally {
      button.disabled = false;
    }
  });
})();
