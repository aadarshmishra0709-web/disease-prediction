(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const boxes = Array.from(document.querySelectorAll('input[name="symptom"]'));
  const labelOf = (cb) => cb.parentElement.querySelector("span").textContent;

  const form = $("predict-form"), errorBox = $("error-box"), btn = $("predict-btn");
  const spinner = btn.querySelector(".spinner"), btnText = btn.querySelector(".btn-text");

  function selected() { return boxes.filter((b) => b.checked); }

  function renderChips() {
    const sel = selected(), chips = $("selected-chips");
    chips.textContent = "";
    sel.forEach((cb) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.append(labelOf(cb));
      const x = document.createElement("button");
      x.type = "button"; x.setAttribute("aria-label", "Remove " + labelOf(cb)); x.textContent = "×";
      x.addEventListener("click", () => { cb.checked = false; renderChips(); });
      chip.append(x); chips.append(chip);
    });
    $("sym-counter").textContent = sel.length + " selected";
  }

  boxes.forEach((b) => b.addEventListener("change", renderChips));

  $("symptom-search").addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    let shown = 0;
    document.querySelectorAll(".symptom").forEach((el) => {
      const match = !q || el.dataset.label.includes(q);
      el.hidden = !match; if (match) shown++;
    });
    $("no-match").hidden = shown !== 0;
  });

  function showError(msg) { errorBox.textContent = msg; errorBox.hidden = false; }
  function clearError() { errorBox.hidden = true; errorBox.textContent = ""; }
  function setLoading(on) {
    btn.disabled = on; spinner.hidden = !on; btnText.textContent = on ? "Predicting…" : "Predict";
  }
  const pct = (x) => (x * 100).toFixed(2) + "%";

  function renderResult(d) {
    $("res-disease").textContent = d.prediction;
    $("res-conf").textContent = pct(d.confidence);
    const badge = $("res-level");
    badge.textContent = d.confidence_level + " confidence";
    badge.className = "badge " + d.confidence_level;

    const warn = $("res-warnings"); warn.textContent = "";
    (d.warnings || []).forEach((w) => {
      const p = document.createElement("div"); p.className = "alert alert-warn"; p.textContent = w; warn.append(p);
    });

    const probs = $("res-probs"); probs.textContent = "";
    Object.entries(d.class_probabilities).forEach(([name, p]) => {
      const row = document.createElement("div"); row.className = "prob-row";
      const n = document.createElement("span"); n.textContent = name;
      const bar = document.createElement("div"); bar.className = "bar";
      const fill = document.createElement("div"); fill.style.width = (p * 100).toFixed(1) + "%"; bar.append(fill);
      const v = document.createElement("span"); v.textContent = pct(p);
      row.append(n, bar, v); probs.append(row);
    });

    const tbody = $("res-models").querySelector("tbody"); tbody.textContent = "";
    Object.entries(d.model_predictions).forEach(([m, r]) => {
      const tr = document.createElement("tr");
      [m, r.prediction, pct(r.confidence)].forEach((t) => { const td = document.createElement("td"); td.textContent = t; tr.append(td); });
      tbody.append(tr);
    });

    const parts = [];
    if (form.dataset.patient) parts.push("Patient details provided (" + form.dataset.patient + ") were not used by the model.");
    $("res-patient").textContent = parts.join(" ");
    $("result").hidden = false;
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault(); clearError();
    const symptoms = selected().map((b) => b.value);
    if (!symptoms.length) { showError("Please select at least one symptom."); return; }

    const payload = { symptoms, age: $("age").value, height: $("height").value, weight: $("weight").value, gender: $("gender").value };
    const given = ["age", "gender", "height", "weight"].filter((k) => payload[k] !== "");
    form.dataset.patient = given.join(", ");

    setLoading(true);
    try {
      const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON response */ }
      if (!res.ok) { showError((data && data.error) || "Server error (" + res.status + ")."); $("result").hidden = true; return; }
      renderResult(data);
    } catch (_) {
      showError("Could not reach the server. Is the app still running?");
    } finally { setLoading(false); }
  });

  $("reset-btn").addEventListener("click", () => {
    form.reset(); boxes.forEach((b) => (b.checked = false));
    $("symptom-search").dispatchEvent(new Event("input"));
    renderChips(); clearError(); $("result").hidden = true; delete form.dataset.patient;
  });

  renderChips();
})();
