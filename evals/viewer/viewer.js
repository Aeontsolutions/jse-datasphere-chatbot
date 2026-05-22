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
  const roster = state.runs.length > 1 ? renderRoster() : "";
  if (state.activeView === "overview") { renderOverview(); }
  else if (state.activeView === "list") { renderList(); }
  else if (state.activeView === "detail") { renderDetail(); }
  else if (state.activeView === "compare") { renderCompare(roster); return; }
  else if (state.activeView === "diff") { renderDiff(roster); return; }
  if (roster && state.activeView !== "compare" && state.activeView !== "diff") {
    viewRoot.insertAdjacentHTML("afterbegin", roster);
  }
}

const DIMENSIONS = [
  "groundedness",
  "factfulness",
  "goal_completion",
  "tool_use_appropriateness",
  "coherence",
  "persona_handling",
];

function renderOverview() {
  const run = state.runs[0];
  const s = run.summary;
  const ov = s.overall;
  viewRoot.innerHTML = `
    <h2>${run.runId}</h2>
    <p><strong>git:</strong> <code>${run.manifest.git_sha || "(unknown)"}</code>
       · <strong>started:</strong> ${run.manifest.started_at}
       · <strong>conversations:</strong> ${s.conversation_count}
       · <strong>total cost:</strong> $${(ov.total_cost_usd || 0).toFixed(4)}
       · <strong>mean latency:</strong> ${Math.round(ov.mean_latency_ms || 0)} ms</p>

    <h3>Verdict mix</h3>
    <p>
      <span class="verdict-pass">pass: ${ov.verdict_counts?.pass || 0}</span> ·
      <span class="verdict-partial">partial: ${ov.verdict_counts?.partial || 0}</span> ·
      <span class="verdict-fail">fail: ${ov.verdict_counts?.fail || 0}</span>
      ${ov.judge_failed_count ? `· <span class="verdict-judgefailed">judge_failed: ${ov.judge_failed_count}</span>` : ""}
    </p>

    <h3>Per-persona</h3>
    ${renderPersonaTable(s.by_persona)}
  `;
}

function renderPersonaTable(byPersona) {
  const personas = Object.keys(byPersona);
  if (!personas.length) return "<p>(no personas)</p>";
  const header = ["persona", "count", ...DIMENSIONS.map(d => d.replace(/_/g, " ")), "verdicts"];
  const rows = personas.map(pid => {
    const p = byPersona[pid];
    return `<tr>
      <td>${pid}</td>
      <td>${p.count}</td>
      ${DIMENSIONS.map(d => `<td>${fmtMeanStd(p[`mean_${d}`], p[`std_${d}`])}</td>`).join("")}
      <td>${fmtVerdicts(p.verdict_counts)}</td>
    </tr>`;
  });
  return `<table>
    <thead><tr>${header.map(h => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}

function fmtMeanStd(mean, std) {
  if (mean == null) return "—";
  if (std == null) return mean.toFixed(2);
  return `${mean.toFixed(2)} ± ${std.toFixed(2)}`;
}

function fmtVerdicts(counts) {
  if (!counts) return "—";
  return `<span class="verdict-pass">${counts.pass || 0}</span>/` +
         `<span class="verdict-partial">${counts.partial || 0}</span>/` +
         `<span class="verdict-fail">${counts.fail || 0}</span>`;
}

function renderList() {
  const run = state.runs[0];
  const rows = run.conversations.map(c => {
    const verdict = c.judge?.verdict || (c.judge?.judge_failed ? "judge_failed" : "unknown");
    return `<tr class="row-clickable" data-id="${c.conversation_id}">
      <td>${c.conversation_id}</td>
      <td>${c.persona?.id || ""}</td>
      <td>${c.endpoint}</td>
      <td class="verdict-${verdict.replace("_", "")}">${verdict}</td>
      <td>${c.totals?.turns ?? "—"}</td>
      <td>${Math.round(c.totals?.latency_ms || 0)} ms</td>
      <td>$${(c.totals?.cost_usd || 0).toFixed(4)}</td>
    </tr>`;
  });
  viewRoot.innerHTML = `<table>
    <thead><tr><th>conversation</th><th>persona</th><th>endpoint</th>
      <th>verdict</th><th>turns</th><th>latency</th><th>cost</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
  viewRoot.querySelectorAll(".row-clickable").forEach(tr => {
    tr.addEventListener("click", () => {
      state.activeConvoId = tr.dataset.id;
      state.activeView = "detail";
      renderApp();
    });
  });
}

function renderDetail() {
  const run = state.runs[0];
  const c = state.activeConvoId
    ? run.conversations.find(x => x.conversation_id === state.activeConvoId)
    : run.conversations[0];
  if (!c) { viewRoot.innerHTML = "<p>(no conversation selected)</p>"; return; }
  const transcript = c.turns.map(renderTurn).join("");
  const judge = c.judge?.judge_failed
    ? `<p class="verdict-judgefailed">Judge failed: ${c.judge.error}</p>`
    : renderJudge(c.judge);
  viewRoot.innerHTML = `
    <h2>${c.conversation_id}</h2>
    <p>endpoint: ${c.endpoint} · termination: ${c.termination.reason} (turn ${c.termination.at_turn})</p>
    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 1rem;">
      <div>${transcript}</div>
      <div>${judge}</div>
    </div>
  `;
  c.turns.forEach((t, i) => {
    const chartSpec = t.chatbot_metadata?.chart?.vega_lite;
    if (chartSpec) {
      vegaEmbed(`#chart-turn-${i}`, chartSpec, { actions: false });
    }
  });
}

function renderTurn(t, i) {
  const chartSlot = t.chatbot_metadata?.chart?.vega_lite
    ? `<div id="chart-turn-${i}" style="margin-top: 0.5rem;"></div>` : "";
  const tools = (t.chatbot_metadata?.tools_executed || []).join(", ");
  const sources = t.chatbot_metadata?.sources || [];
  const filters = t.chatbot_metadata?.filters_used;
  const drawer = `
    <details class="turn-drawer">
      <summary>raw metadata</summary>
      ${sources.length ? `<h5>sources</h5><ul>${sources.map(s =>
        `<li>${escapeHtml(s.title || JSON.stringify(s))}</li>`
      ).join("")}</ul>` : ""}
      ${filters ? `<h5>filters_used</h5><pre>${escapeHtml(JSON.stringify(filters, null, 2))}</pre>` : ""}
      <h5>full response</h5>
      <pre>${escapeHtml(JSON.stringify(t.chatbot_metadata, null, 2))}</pre>
    </details>
  `;
  return `
    <div class="turn-row">
      <div class="bubble user">${escapeHtml(t.persona_utterance)}</div>
      <div class="bubble bot">${escapeHtml(t.chatbot_text)}${chartSlot}</div>
      <div class="turn-meta">turn ${t.turn_index} · ${Math.round(t.latency_ms)} ms${t.cost_usd ? ` · $${t.cost_usd.toFixed(4)}` : ""}${tools ? ` · tools: ${tools}` : ""}</div>
      ${drawer}
    </div>
  `;
}

function renderJudge(j) {
  if (!j) return "<p>(not judged)</p>";
  const scoreRows = DIMENSIONS.map(d => {
    const s = j.scores?.[d];
    if (!s) return "";
    const score = s.score == null ? "—" : s.score;
    return `<tr><td>${d.replace(/_/g, " ")}</td><td>${score}</td><td>${escapeHtml(s.justification || "")}</td></tr>`;
  }).join("");
  const notable = (j.notable_moments || []).map(m =>
    `<li>turn ${m.turn} (${m.type}): ${escapeHtml(m.note)}</li>`
  ).join("");
  return `
    <h3 class="verdict-${j.verdict}">verdict: ${j.verdict}</h3>
    <p>${escapeHtml(j.verdict_reason || "")}</p>
    <table>
      <thead><tr><th>dimension</th><th>score</th><th>justification</th></tr></thead>
      <tbody>${scoreRows}</tbody>
    </table>
    ${notable ? `<h4>Notable moments</h4><ul>${notable}</ul>` : ""}
  `;
}

function renderDiff(roster) {
  const a = state.diff.a ? state.runs.find(r => r.runId === state.diff.a) : state.runs[0];
  const b = state.diff.b ? state.runs.find(r => r.runId === state.diff.b) : state.runs[1];
  if (!a || !b) { viewRoot.innerHTML = roster + "<p>Need two runs loaded.</p>"; return; }

  const pickers = `
    <p>
      A (baseline): <select id="diff-a">${state.runs.map(r => `<option value="${r.runId}"${r.runId === a.runId ? " selected" : ""}>${r.runId}</option>`).join("")}</select>
      &nbsp;B (candidate): <select id="diff-b">${state.runs.map(r => `<option value="${r.runId}"${r.runId === b.runId ? " selected" : ""}>${r.runId}</option>`).join("")}</select>
    </p>
  `;

  const personas = new Set([
    ...Object.keys(a.summary.by_persona || {}),
    ...Object.keys(b.summary.by_persona || {}),
  ]);

  // Per-persona delta cells
  const rows = [...personas].map(pid => {
    const ap = a.summary.by_persona[pid];
    const bp = b.summary.by_persona[pid];
    const cells = [pid];
    DIMENSIONS.forEach(d => {
      const aMean = ap?.[`mean_${d}`];
      const bMean = bp?.[`mean_${d}`];
      const aStd = ap?.[`std_${d}`] || 0;
      const bStd = bp?.[`std_${d}`] || 0;
      if (aMean == null || bMean == null) { cells.push("—"); return; }
      const delta = bMean - aMean;
      const noise = Math.max(0.5, aStd + bStd);
      const cls = Math.abs(delta) < noise ? "delta-noise" : (delta > 0 ? "delta-up" : "delta-down");
      cells.push(`<span class="${cls}">${delta >= 0 ? "+" : ""}${delta.toFixed(2)}</span>`);
    });
    return `<tr><td>${pid}</td>${cells.slice(1).map(c => `<td>${c}</td>`).join("")}</tr>`;
  });

  // Top movers across all (persona × dimension) cells
  const movers = [];
  for (const pid of personas) {
    for (const d of DIMENSIONS) {
      const aMean = a.summary.by_persona[pid]?.[`mean_${d}`];
      const bMean = b.summary.by_persona[pid]?.[`mean_${d}`];
      if (aMean == null || bMean == null) continue;
      movers.push({ pid, dim: d, delta: bMean - aMean });
    }
  }
  movers.sort((x, y) => y.delta - x.delta);
  const top = movers.slice(0, 5);
  const bottom = movers.slice(-5).reverse();

  const pairOptions = [...personas].map(p => `<option value="${p}">${p}</option>`).join("");

  viewRoot.innerHTML = `
    ${roster}
    ${pickers}
    <h3>Per-persona dimension deltas (B − A)</h3>
    <table>
      <thead><tr><th>persona</th>${DIMENSIONS.map(d => `<th>${d.replace(/_/g, " ")}</th>`).join("")}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>

    <h3>Biggest improvements</h3>
    <ol>${top.map(m => `<li><strong>${m.pid}</strong> · ${m.dim}: <span class="delta-up">+${m.delta.toFixed(2)}</span></li>`).join("")}</ol>

    <h3>Biggest regressions</h3>
    <ol>${bottom.map(m => `<li><strong>${m.pid}</strong> · ${m.dim}: <span class="delta-down">${m.delta.toFixed(2)}</span></li>`).join("")}</ol>

    <h3>Conversation pair</h3>
    <p>Persona: <select id="pair-persona">${pairOptions}</select>
       Replicate: <input id="pair-rep" type="number" min="1" max="20" value="1" /></p>
    <div id="pair-display"></div>
  `;

  document.getElementById("diff-a").onchange = e => { state.diff.a = e.target.value; renderApp(); };
  document.getElementById("diff-b").onchange = e => { state.diff.b = e.target.value; renderApp(); };

  function renderPair() {
    const pid = document.getElementById("pair-persona").value;
    const rep = parseInt(document.getElementById("pair-rep").value, 10) || 1;
    const cid = `${pid}__rep${String(rep).padStart(2, "0")}`;
    const aC = a.conversations.find(c => c.conversation_id === cid);
    const bC = b.conversations.find(c => c.conversation_id === cid);
    const pairEl = document.getElementById("pair-display");
    pairEl.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div><h4>A: ${a.runId}</h4>${aC ? aC.turns.map(renderTurn).join("") : "<p>(not found)</p>"}
          ${aC?.judge ? renderJudge(aC.judge) : ""}</div>
        <div><h4>B: ${b.runId}</h4>${bC ? bC.turns.map(renderTurn).join("") : "<p>(not found)</p>"}
          ${bC?.judge ? renderJudge(bC.judge) : ""}</div>
      </div>
    `;
  }
  document.getElementById("pair-persona").onchange = renderPair;
  document.getElementById("pair-rep").onchange = renderPair;
  renderPair();
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, c => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
  ));
}

// ---------- Run roster rendering (multi-run mode) ----------

function renderRoster() {
  if (state.runs.length < 2) return "";
  const baselineId = state.diff.a || state.runs[0].runId;
  const candidateId = state.diff.b || (state.runs[1] && state.runs[1].runId);
  const rows = state.runs.map(r => {
    const role = r.runId === baselineId ? "baseline" :
                 r.runId === candidateId ? "candidate" : "loaded";
    return `<tr>
      <td>${r.runId}</td>
      <td><code>${r.manifest.git_sha || "—"}</code></td>
      <td>${r.manifest.started_at}</td>
      <td>${r.summary.conversation_count}</td>
      <td>$${(r.summary.overall.total_cost_usd || 0).toFixed(4)}</td>
      <td>${role}</td>
    </tr>`;
  });
  return `<h3>Loaded runs (${state.runs.length})</h3>
    <table>
      <thead><tr><th>run id</th><th>git</th><th>started</th><th>convos</th><th>cost</th><th>role</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;
}

function renderCompare(roster) {
  // Union of personas across all runs
  const allPersonas = new Set();
  state.runs.forEach(r => Object.keys(r.summary.by_persona || {}).forEach(p => allPersonas.add(p)));

  const baseline = state.runs[0];
  const scorecards = state.runs.map(r => {
    const ov = r.summary.overall;
    return `<div style="border: 1px solid var(--border); padding: 0.75rem; border-radius: 0.5rem;">
      <h4>${r.runId} ${r.runId === baseline.runId ? "<small>(baseline)</small>" : ""}</h4>
      <p>convos: ${r.summary.conversation_count} · cost: $${(ov.total_cost_usd || 0).toFixed(4)} · turns: ${(ov.mean_turns || 0).toFixed(1)}</p>
      <p>
        <span class="verdict-pass">${ov.verdict_counts?.pass || 0}</span>/
        <span class="verdict-partial">${ov.verdict_counts?.partial || 0}</span>/
        <span class="verdict-fail">${ov.verdict_counts?.fail || 0}</span>
      </p>
    </div>`;
  });

  // Per-persona table: rows = personas; columns grouped by run
  const headerCells = ["persona"];
  state.runs.forEach(r => DIMENSIONS.forEach(d => headerCells.push(`${r.runId.slice(0, 8)}<br><small>${d}</small>`)));
  const rows = [...allPersonas].map(pid => {
    const cells = [pid];
    state.runs.forEach(r => {
      const p = r.summary.by_persona?.[pid];
      DIMENSIONS.forEach(d => {
        const mean = p?.[`mean_${d}`];
        const std = p?.[`std_${d}`];
        const cell = mean == null ? "—" : fmtMeanStd(mean, std);
        // delta vs baseline
        if (r.runId !== baseline.runId) {
          const bMean = baseline.summary.by_persona?.[pid]?.[`mean_${d}`];
          const bStd = baseline.summary.by_persona?.[pid]?.[`std_${d}`];
          if (mean != null && bMean != null) {
            const delta = mean - bMean;
            const noise = Math.max(0.5, (std || 0) + (bStd || 0));
            const cls = Math.abs(delta) < noise ? "delta-noise"
                       : (delta > 0 ? "delta-up" : "delta-down");
            cells.push(`${cell}<br><span class="${cls}">Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(2)}</span>`);
            return;
          }
        }
        cells.push(cell);
      });
    });
    return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
  });

  viewRoot.innerHTML = `
    ${roster}
    <h3>Side-by-side scorecards</h3>
    <div style="display: grid; grid-template-columns: repeat(${state.runs.length}, 1fr); gap: 1rem;">
      ${scorecards.join("")}
    </div>
    <h3>Per-persona dimensions vs baseline (${baseline.runId})</h3>
    <table>
      <thead><tr>${headerCells.map(h => `<th>${h}</th>`).join("")}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

// ---------- Auto-load from query string ----------

async function autoLoadFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const single = params.get("run");
  const many = params.get("runs");
  const ids = single ? [single] : (many ? many.split(",") : []);
  for (const id of ids) {
    try {
      const manifest = await (await fetch(`../runs/${id}/manifest.json`)).json();
      const summary = await (await fetch(`../runs/${id}/summary.json`)).json();
      // Fetch conversation index by listing — best effort using manifest.personas_run × replicates
      const conversations = [];
      for (const p of manifest.personas_run || []) {
        for (let r = 1; r <= (manifest.replicates || 1); r++) {
          const cid = `${p}__rep${String(r).padStart(2, "0")}`;
          try {
            const c = await (await fetch(`../runs/${id}/conversations/${cid}.json`)).json();
            conversations.push(c);
          } catch (_e) { /* missing — skip */ }
        }
      }
      state.runs.push({ runId: manifest.run_id || id, manifest, summary, conversations });
    } catch (e) {
      setStatus(`Failed to load ${id}: ${e.message}`);
    }
  }
  if (state.runs.length) renderApp();
}

autoLoadFromQuery();
