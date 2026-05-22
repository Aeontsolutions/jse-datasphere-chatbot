"use strict";

const state = {
  runs: [],            // [{ runId, manifest, summary, conversations: [...] }]
  activeView: null,    // 'overview' | 'list' | 'detail' | 'compare' | 'diff'
  activeConvoId: null, // for detail view
  diff: { a: null, b: null },
};

// ---------- File loading ----------

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const viewRoot = document.getElementById("view-root");

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", async e => {
  const items = Array.from(e.dataTransfer.items || []);
  const files = [];
  for (const item of items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) await collectEntryFiles(entry, files);
  }
  if (files.length) await ingestFiles(files);
});
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files);
  if (files.length) await ingestFiles(files);
});

async function collectEntryFiles(entry, out, basePath = "") {
  if (entry.isFile) {
    const file = await new Promise(res => entry.file(res));
    file._relPath = basePath + entry.name;
    out.push(file);
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    const entries = await new Promise(res => reader.readEntries(res));
    for (const sub of entries) {
      await collectEntryFiles(sub, out, basePath + entry.name + "/");
    }
  }
}

async function ingestFiles(files) {
  const runs = bucketByRun(files);
  for (const [runId, fileMap] of Object.entries(runs)) {
    const run = await parseRun(runId, fileMap);
    if (run) state.runs.push(run);
  }
  renderApp();
}

function bucketByRun(files) {
  // Detect run roots by finding manifest.json files
  const runs = {};
  for (const f of files) {
    const rel = f._relPath || f.webkitRelativePath || f.name;
    const parts = rel.split("/");
    // Simpler heuristic: the first path segment under the dropped root is the run id
    const runId = parts[0];
    runs[runId] ||= {};
    runs[runId][rel] = f;
  }
  return runs;
}

async function parseRun(runId, fileMap) {
  const manifestKey = Object.keys(fileMap).find(k => k.endsWith("manifest.json"));
  const summaryKey = Object.keys(fileMap).find(k => k.endsWith("summary.json"));
  if (!manifestKey || !summaryKey) {
    setStatus(`Skipped ${runId}: no manifest.json/summary.json`);
    return null;
  }
  const manifest = await readJson(fileMap[manifestKey]);
  const summary = await readJson(fileMap[summaryKey]);
  const convoFiles = Object.entries(fileMap).filter(([k]) => k.includes("/conversations/"));
  const conversations = await Promise.all(convoFiles.map(([, f]) => readJson(f)));
  return { runId: manifest.run_id || runId, manifest, summary, conversations };
}

function readJson(file) {
  return file.text().then(t => JSON.parse(t));
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

// ---------- Rendering ----------

function renderApp() {
  renderTabs();
  if (!state.activeView) {
    state.activeView = state.runs.length > 1 ? "compare" : "overview";
  }
  renderActiveView();
}

function renderTabs() {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = "";
  const opts = state.runs.length > 1
    ? ["compare", "diff", "list", "detail"]
    : ["overview", "list", "detail"];
  for (const opt of opts) {
    const btn = document.createElement("button");
    btn.textContent = opt;
    btn.className = state.activeView === opt ? "active" : "";
    btn.onclick = () => {
      state.activeView = opt;
      renderApp();
    };
    tabs.appendChild(btn);
  }
}

function renderActiveView() {
  if (!state.runs.length) {
    viewRoot.innerHTML = "<p>No runs loaded yet.</p>";
    return;
  }
  if (state.activeView === "overview") renderOverview();
  else viewRoot.innerHTML = `<p>(${state.activeView} view not implemented yet)</p>`;
}

function renderOverview() {
  const run = state.runs[0];
  const s = run.summary;
  viewRoot.innerHTML = `
    <h2>${run.runId}</h2>
    <p>git: <code>${run.manifest.git_sha || "(unknown)"}</code> · conversations: ${s.conversation_count} · total cost: $${(s.overall.total_cost_usd || 0).toFixed(4)}</p>
    <p>(Overview table coming in next task)</p>
  `;
}
