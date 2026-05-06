import { WebViewer } from "https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@0.21.0/+esm";

const state = {
  objects: [],
  removedTargets: new Set(),
  selectedEntity: null,
  selectedObject: null,
  viewer: null,
};

const els = {
  config: document.querySelector("#config"),
  viewer: document.querySelector("#viewer"),
  objectList: document.querySelector("#object-list"),
  rerunSelection: document.querySelector("#rerun-selection"),
  addSelected: document.querySelector("#add-selected"),
  clear: document.querySelector("#clear"),
  refresh: document.querySelector("#refresh"),
  remove: document.querySelector("#remove"),
  exportPath: document.querySelector("#export-path"),
  status: document.querySelector("#status"),
};

function setStatus(value) {
  els.status.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}

function extractObjectFromSelection(payload) {
  const text = JSON.stringify(payload);
  const pathMatch = text.match(/(?:^|[/"\\])world[/"\\]+objects[/"\\]+([^/"\\\s]+)/);
  if (pathMatch) return pathMatch[1];

  const arrayPathMatch = text.match(/"world"\s*,\s*"objects"\s*,\s*"([^"]+)"/);
  if (arrayPathMatch) return arrayPathMatch[1];

  const objectMatch = text.match(/objects[/"\\]+([^/"\\\s]+)/);
  if (objectMatch) return objectMatch[1];

  const labelMatch = text.match(/([0-9]+_[a-z0-9_-]+)\s*·/i);
  if (labelMatch) return labelMatch[1];

  return state.objects.find((obj) => text.includes(obj.folder))?.folder || null;
}

function renderObjects() {
  els.objectList.innerHTML = "";
  for (const obj of state.objects) {
    if (state.removedTargets.has(obj.folder)) continue;

    const row = document.createElement("label");
    row.className = "object-row";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = obj.folder;

    const name = document.createElement("span");
    name.title = obj.folder;
    name.textContent = `${obj.folder} · ${obj.label || ""}`;

    const count = document.createElement("span");
    count.className = "count";
    count.textContent = obj.num_points ?? "";

    row.append(box, name, count);
    els.objectList.append(row);
  }
}

function checkedTargets() {
  return [...els.objectList.querySelectorAll("input:checked")].map((input) => input.value);
}

function checkTarget(folder) {
  const input = els.objectList.querySelector(`input[value="${CSS.escape(folder)}"]`);
  if (!input) return false;
  input.checked = true;
  input.dispatchEvent(new Event("change", { bubbles: true }));
  input.closest(".object-row")?.classList.add("selected");
  return true;
}

async function loadRecording(recordingUrl) {
  const url = new URL(recordingUrl, window.location.origin).href;
  els.viewer.innerHTML = "";
  state.viewer = new WebViewer();
  await state.viewer.start(url, els.viewer, { width: "100%", height: "100%" });
  state.viewer.on("selection_change", (...payload) => {
    state.selectedEntity = payload;
    state.selectedObject = extractObjectFromSelection(payload);
    els.rerunSelection.textContent = state.selectedObject || JSON.stringify(payload);
    if (state.selectedObject) {
      const checked = checkTarget(state.selectedObject);
      if (!checked) {
        setStatus(`Selected ${state.selectedObject}, but it is not in the current object list.`);
      }
    } else {
      setStatus(`Could not parse Rerun selection:\n${JSON.stringify(payload, null, 2)}`);
    }
  });
}

async function init() {
  const config = await getJson("/api/config");
  els.config.textContent = `${config.mode}\n${config.point_cloud}\n${config.layout}`;

  const objects = await getJson("/api/objects");
  state.objects = objects.objects;
  renderObjects();

  const recording = await getJson("/api/recording");
  await loadRecording(recording.url);

  setStatus("Ready.");
}

els.addSelected.addEventListener("click", () => {
  if (!state.selectedObject) {
    setStatus("No object path found in current Rerun selection.");
    return;
  }
  checkTarget(state.selectedObject);
});

els.clear.addEventListener("click", () => {
  for (const input of els.objectList.querySelectorAll("input")) input.checked = false;
});

els.refresh.addEventListener("click", () => window.location.reload());

els.remove.addEventListener("click", async () => {
  const currentTargets = checkedTargets();
  if (!currentTargets.length) {
    setStatus("Select at least one object.");
    return;
  }
  const targets = [...new Set([...state.removedTargets, ...currentTargets])];
  setStatus(`Exporting removed scene for ${targets.length} cumulative target(s)...`);
  try {
    const report = await postJson("/api/remove", { targets });
    if (report.recording_url) {
      await loadRecording(report.recording_url);
    }
    for (const target of currentTargets) state.removedTargets.add(target);
    renderObjects();
    for (const input of els.objectList.querySelectorAll("input")) input.checked = false;
    state.selectedEntity = null;
    state.selectedObject = null;
    els.rerunSelection.textContent = "None";
    els.exportPath.textContent = report.output_ply || "No export path returned.";
    setStatus(report);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
});

init().catch((error) => setStatus(`Error: ${error.message}`));
