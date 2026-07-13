const state = {
  codes: [],
  decisions: {},
  filtered: [],
  index: 0,
  selectedDecision: null,
  summary: null,
  cheatSheets: [],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const decisionLabels = {
  approve: "Código e grupo aprovados",
  remap: "Remapear grupo",
  mixed: "Código misto ou incorreto",
  quarantine: "Quarentena",
  ambiguous: "Ambíguo",
};

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.style.background = error ? "#a62f2f" : "#142438";
  element.classList.add("visible");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("visible"), 2400);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function reviewer() {
  return $("#reviewer").value.trim();
}

function currentCode() {
  return state.filtered[state.index] || null;
}

function decisionFor(code) {
  return code ? state.decisions[code.source_code] : null;
}

function groupLabel(group) {
  return group === "quarantine_traffic_light" ? "quarentena · semáforo" : group;
}

function applyFilters(preserveCode = null) {
  const status = $("#status-filter").value;
  const search = $("#search-filter").value.trim().toLowerCase();
  state.filtered = state.codes.filter((code) => {
    const record = decisionFor(code);
    const searchable = [code.source_code, code.official_code, code.expected_name, code.current_group]
      .join(" ")
      .toLowerCase();
    if (search && !searchable.includes(search)) return false;
    if (status === "pending" && record) return false;
    if (status === "reviewed" && !record) return false;
    if (status && !["pending", "reviewed"].includes(status) && record?.decision !== status) return false;
    return true;
  });
  const preserved = preserveCode
    ? state.filtered.findIndex((code) => code.source_code === preserveCode)
    : -1;
  state.index = preserved >= 0
    ? preserved
    : Math.min(state.index, Math.max(0, state.filtered.length - 1));
  $("#filter-count").textContent = `${state.filtered.length} código(s) no filtro`;
  renderCurrent();
}

function updateProgress() {
  const summary = state.summary;
  const pct = summary.total ? (summary.reviewed / summary.total) * 100 : 0;
  $("#progress-text").textContent = `${summary.reviewed} de ${summary.total} códigos revisados · ${summary.total_occurrences} ocorrências disponíveis`;
  $("#progress-bar").style.width = `${pct}%`;
  $("#finalize").disabled = summary.remaining !== 0;
}

function setSelectedDecision(value) {
  state.selectedDecision = value;
  $$('[data-decision]').forEach((button) => {
    button.classList.toggle("selected", button.dataset.decision === value);
    button.setAttribute("aria-checked", String(button.dataset.decision === value));
  });
  $("#corrected-class").disabled = value !== "remap";
  if (value !== "remap") $("#corrected-class").value = "";
}

function openOccurrence(item) {
  $("#occurrence-dialog-title").textContent = `${item.image_id} · box ${item.box_index}`;
  $("#occurrence-large").src = item.crop_url;
  $("#occurrence-metadata").textContent = `Código ${item.source_code} · bbox ${item.bbox_xyxy.join(", ")}`;
  $("#occurrence-dialog").showModal();
}

function renderOccurrences(code) {
  const grid = $("#occurrence-grid");
  grid.innerHTML = "";
  if (!code) {
    grid.innerHTML = '<p class="empty">Nenhum código corresponde ao filtro.</p>';
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const item of code.occurrences) {
    const button = document.createElement("button");
    button.className = "occurrence-card";
    button.type = "button";
    button.title = `Ampliar ${item.image_id}, box ${item.box_index}`;
    button.addEventListener("click", () => openOccurrence(item));

    const image = document.createElement("img");
    image.src = item.crop_url;
    image.alt = `Recorte ${item.image_id}, box ${item.box_index}`;
    image.loading = "lazy";

    const label = document.createElement("span");
    label.textContent = `${item.image_id} · ${item.box_index}`;
    button.append(image, label);
    fragment.appendChild(button);
  }
  grid.appendChild(fragment);
}

function renderCurrent() {
  const code = currentCode();
  $("#previous").disabled = state.index <= 0;
  $("#next").disabled = state.index >= state.filtered.length - 1;
  if (!code) {
    $("#code-title").textContent = "Nenhum código no filtro";
    $("#code-meaning").textContent = "Ajuste os filtros para continuar.";
    $("#current-group").textContent = "—";
    $("#occurrence-count").textContent = "—";
    $("#position").textContent = "—";
    $("#note").value = "";
    setSelectedDecision(null);
    renderOccurrences(null);
    return;
  }
  $("#code-title").textContent = `${code.source_code} · ${code.official_code}`;
  $("#code-meaning").textContent = code.expected_name;
  $("#current-group").textContent = groupLabel(code.current_group);
  $("#occurrence-count").textContent = String(code.occurrence_count);
  $("#position").textContent = `${state.index + 1} / ${state.filtered.length}`;
  const record = decisionFor(code);
  setSelectedDecision(record?.decision || null);
  $("#corrected-class").value = record?.corrected_class || "";
  $("#note").value = record?.note || "";
  $("#occurrence-help").textContent = `${code.occurrence_count} boxes anotadas como ${code.source_code}. Clique em uma miniatura para ampliar.`;
  renderOccurrences(code);
}

function buildCheatSheets() {
  const gallery = $("#sheet-gallery");
  gallery.innerHTML = "";
  for (const sheet of state.cheatSheets) {
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    caption.textContent = sheet.title;
    const image = document.createElement("img");
    image.src = sheet.image_url;
    image.alt = sheet.title;
    figure.append(caption, image);
    gallery.appendChild(figure);
  }
}

function openCheatSheet() {
  const dialog = $("#cheatsheet-dialog");
  dialog.showModal();
  dialog.scrollTop = 0;
}

async function saveDecision() {
  const code = currentCode();
  if (!code) return;
  if (!reviewer()) return toast("Informe e confirme o nome do revisor.", true);
  if (!state.selectedDecision) return toast("Escolha uma decisão para o código.", true);
  const correctedClass = $("#corrected-class").value;
  if (state.selectedDecision === "remap" && !correctedClass) {
    return toast("Selecione o novo grupo operacional.", true);
  }
  $("#save-status").textContent = "Salvando…";
  try {
    const result = await request("/api/review", {
      method: "POST",
      body: JSON.stringify({
        source_code: code.source_code,
        decision: state.selectedDecision,
        corrected_class: correctedClass,
        note: $("#note").value,
        reviewer: reviewer(),
      }),
    });
    state.decisions[code.source_code] = result.record;
    state.summary = result.summary;
    updateProgress();
    $("#save-status").textContent = `Salvo · ${new Date().toLocaleTimeString()}`;
    toast(`${decisionLabels[state.selectedDecision]}: ${code.source_code}.`);
    if ($("#status-filter").value === "pending") {
      applyFilters();
    } else if (state.index < state.filtered.length - 1) {
      state.index += 1;
      renderCurrent();
    } else {
      renderCurrent();
    }
  } catch (error) {
    $("#save-status").textContent = "Erro ao salvar";
    toast(error.message, true);
  }
}

async function initialize() {
  const payload = await request("/api/state");
  state.codes = payload.codes;
  state.decisions = payload.decisions;
  state.summary = payload.summary;
  state.cheatSheets = payload.cheat_sheets;
  buildCheatSheets();
  const savedReviewer = localStorage.getItem("bvtsld-reviewer") || payload.review.reviewers.at(-1) || "";
  $("#reviewer").value = savedReviewer;
  updateProgress();
  applyFilters();
}

$$('[data-decision]').forEach((button) => button.addEventListener("click", () => setSelectedDecision(button.dataset.decision)));
$("#save").addEventListener("click", saveDecision);
$("#previous").addEventListener("click", () => { state.index = Math.max(0, state.index - 1); renderCurrent(); });
$("#next").addEventListener("click", () => { state.index = Math.min(state.filtered.length - 1, state.index + 1); renderCurrent(); });
$("#status-filter").addEventListener("change", () => { state.index = 0; applyFilters(); });
$("#search-filter").addEventListener("input", () => { state.index = 0; applyFilters(); });
$("#set-reviewer").addEventListener("click", async () => {
  try {
    await request("/api/reviewer", { method: "POST", body: JSON.stringify({ reviewer: reviewer() }) });
    localStorage.setItem("bvtsld-reviewer", reviewer());
    toast("Revisor confirmado.");
  } catch (error) { toast(error.message, true); }
});
$("#cheatsheet-open").addEventListener("click", openCheatSheet);
$("#cheatsheet-inline").addEventListener("click", openCheatSheet);
$("#cheatsheet-close").addEventListener("click", () => $("#cheatsheet-dialog").close());
$("#occurrence-close").addEventListener("click", () => $("#occurrence-dialog").close());
$("#finalize").addEventListener("click", async () => {
  if (!confirm("Finalizar a auditoria por código e marcar o relatório como human_approved?")) return;
  try {
    await request("/api/finalize", { method: "POST", body: JSON.stringify({ reviewer: reviewer() }) });
    toast("Auditoria por código finalizada.");
  } catch (error) { toast(error.message, true); }
});

document.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === "ArrowLeft") $("#previous").click();
  if (event.key === "ArrowRight") $("#next").click();
});

initialize().catch((error) => toast(error.message, true));
