// "Label subjects": place the face boxes over the photo, click a box to cycle its role (background -> main ->
// co-subject), keep the radio buttons in step, and allow only one main face. An uploaded photo is shown from the
// browser's memory (blob URL); the server gets it once for detection and keeps only the boxes and signals.
(function () {
  const CYCLE = { background: "main", main: "co_subject", co_subject: "background" };

  function place(stage) {
    stage.querySelectorAll(".face").forEach((box) => {
      box.style.left = (parseFloat(box.dataset.x) * 100) + "%";
      box.style.top = (parseFloat(box.dataset.y) * 100) + "%";
      box.style.width = (parseFloat(box.dataset.w) * 100) + "%";
      box.style.height = (parseFloat(box.dataset.h) * 100) + "%";
    });
    const ph = stage.querySelector(".placeholder");
    if (ph) ph.style.aspectRatio = ph.dataset.w + " / " + ph.dataset.h;
  }

  function setRole(form, i, role) {
    if (role === "main") {  // only one main face: a previous main becomes co-subject
      form.querySelectorAll('input[value="main"]:checked').forEach((r) => {
        const j = r.name.slice(5);
        if (j !== String(i)) setRole(form, j, "co_subject");
      });
    }
    const radio = form.querySelector(`input[name="role_${i}"][value="${role}"]`);
    if (radio) radio.checked = true;
    const box = form.querySelector(`.face[data-i="${i}"]`);
    if (box) box.className = "face role-" + role;
  }

  function wire(form) {
    const stage = form.querySelector(".stage");
    place(stage);
    stage.addEventListener("click", (ev) => {
      const box = ev.target.closest(".face");
      if (!box) return;
      const i = box.dataset.i;
      const cur = form.querySelector(`input[name="role_${i}"]:checked`);
      setRole(form, i, CYCLE[cur ? cur.value : "background"]);
    });
    form.addEventListener("change", (ev) => {
      const r = ev.target;
      if (r.type === "radio" && r.name.startsWith("role_")) setRole(form, r.name.slice(5), r.value);
    });
  }

  document.querySelectorAll("form.label-form").forEach(wire);

  // upload flow (index page)
  const upload = document.getElementById("subject-upload");
  if (!upload) return;
  const area = document.getElementById("upload-area");
  const msg = document.getElementById("message");
  const tpl = document.getElementById("label-form-template");
  let url = null;

  upload.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const file = upload.querySelector("input[type=file]").files[0];
    if (!file) return;
    const button = upload.querySelector("button");
    button.disabled = true;
    area.replaceChildren();
    msg.textContent = "Detecting faces…";
    msg.className = "notice";
    const body = new FormData();
    body.append("image", file);
    try {
      const res = await fetch("/subjects/upload", {
        method: "POST", body, headers: { "X-CSRF-Token": upload.dataset.csrf }, credentials: "same-origin",
      });
      const data = await res.json();
      if (!res.ok) throw new Error((data.error && data.error.message) || res.statusText);
      if (!data.id) { msg.textContent = "No faces found in that photo."; return; }
      msg.textContent = `${data.faces.length} face(s). Mark the main subject, then save.`;
      if (url) URL.revokeObjectURL(url);
      url = URL.createObjectURL(file);
      const form = tpl.content.firstElementChild.cloneNode(true);
      form.action = "/subjects/" + data.id;
      const stage = form.querySelector(".stage");
      stage.querySelector("img").src = url;
      const roles = form.querySelector(".face-roles");
      data.faces.forEach((f, i) => {
        const box = document.createElement("div");
        box.className = "face role-background";
        Object.assign(box.dataset, { i: String(i), x: f.box.x, y: f.box.y, w: f.box.w, h: f.box.h });
        const n = document.createElement("span");
        n.textContent = String(i + 1);
        box.appendChild(n);
        stage.appendChild(box);
        const fs = document.createElement("fieldset");
        fs.className = "face-role";
        const lg = document.createElement("legend");
        lg.textContent = "Face " + (i + 1);
        fs.appendChild(lg);
        [["main", "Main"], ["co_subject", "Co-subject"], ["background", "Background"]].forEach(([v, t]) => {
          const label = document.createElement("label");
          const r = document.createElement("input");
          Object.assign(r, { type: "radio", name: "role_" + i, value: v, checked: v === "background" });
          label.append(r, " " + t);
          fs.appendChild(label);
        });
        roles.appendChild(fs);
      });
      area.appendChild(form);
      wire(form);
    } catch (err) {
      msg.textContent = "Failed: " + err.message;
      msg.className = "notice error";
    } finally {
      button.disabled = false;
    }
  });
})();
