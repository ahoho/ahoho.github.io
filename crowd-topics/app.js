"use strict";
/* ACL Anthology topic browser — static D3 app over precomputed JSON + binary in data/. */

const App = {
  manifest: null, docsMeta: null, vocab: null,
  K: null, seed: null, tab: "topics", sortMode: "order",
  runs: {},            // runKey -> {topics, topicDocs, topicTime, colorByTid}
  sel: null, selRef: null, lambda: 0.6, tau: null,
  graph: null, sim: null, _yc: null,
};

// dataset root: ?ds=data_crowd loads the crowdsourcing subset; default is the full corpus
const DATA = (new URLSearchParams(location.search).get("ds") || "data").replace(/[^\w/-]/g, "");

/* ---------- utils ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const cache = new Map(), binCache = new Map();
function getJSON(path) {
  if (!cache.has(path)) cache.set(path, fetch(path).then(r => {
    if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json();
  }));
  return cache.get(path);
}
function getBin(path) {
  if (!binCache.has(path)) binCache.set(path, fetch(path).then(r => {
    if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.arrayBuffer();
  }));
  return binCache.get(path);
}
function hexRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}
const pct = x => (100 * x).toFixed(1) + "%";
const esc = s => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const tip = $("#tooltip");
function showTip(html, ev) {
  tip.innerHTML = html; tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth) x = ev.clientX - w - pad;
  if (y + h > innerHeight) y = ev.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };

const runKey = (K = App.K, seed = App.seed) => `k${K}_s${seed}`;
const curRun = () => App.runs[runKey()];
const curTopics = () => curRun().topics.topics;
const topicById = id => curTopics().find(t => t.id === id);

/* ---------- init ---------- */
async function init() {
  [App.manifest, App.docsMeta, App.vocab] = await Promise.all([
    getJSON(`${DATA}/manifest.json`), getJSON(`${DATA}/docs_meta.json`), getJSON(`${DATA}/vocab.json`),
  ]);
  const yr = App.manifest.years;
  $("#subtitle").textContent =
    `${App.manifest.n_docs.toLocaleString()} abstracts · ${yr[0]}–${yr[yr.length - 1]} · LDA (tomotopy)`
    + (App.manifest.has_platform ? " · crowdsourcing subset" : "");
  buildSegs();
  $$("#tabs button").forEach(b => b.onclick = () => selectTab(b.dataset.tab));
  $("#topic-sort").onchange = e => { App.sortMode = e.target.value; renderTopicList(); };
  bindStabilityControls();

  App.K = App.manifest.ks.includes(50) ? 50 : App.manifest.ks[0];
  App.seed = App.manifest.canonical_seed;
  await loadRun(App.K, App.seed);
  updateSegs();
  renderTopicList();
  selectTopic([...curTopics()].sort((a, b) => b.prevalence - a.prevalence)[0].id);
}

function buildSegs() {
  const ks = $("#k-seg"); ks.innerHTML = "";
  for (const k of App.manifest.ks) {
    const b = document.createElement("button"); b.textContent = "K=" + k; b.dataset.k = k;
    b.onclick = () => setRun(k, App.seed); ks.appendChild(b);
  }
  const ss = $("#seed-seg"); ss.innerHTML = "";
  for (const s of App.manifest.seeds) {
    const b = document.createElement("button"); b.textContent = s; b.dataset.seed = s;
    b.title = "random seed " + s; b.onclick = () => setRun(App.K, s); ss.appendChild(b);
  }
}
function updateSegs() {
  $$("#k-seg button").forEach(b => b.classList.toggle("active", +b.dataset.k === App.K));
  $$("#seed-seg button").forEach(b => b.classList.toggle("active", +b.dataset.seed === App.seed));
}

async function loadRun(K, seed) {
  const key = `k${K}_s${seed}`;
  if (App.runs[key]) return;
  const [topics, topicDocs, topicTime] = await Promise.all([
    getJSON(`${DATA}/${key}/topics.json`),
    getJSON(`${DATA}/${key}/topic_docs.json`),
    getJSON(`${DATA}/${key}/topic_time.json`),
  ]);
  const colorByTid = {}; for (const t of topics.topics) colorByTid[t.id] = t.color;
  App.runs[key] = { topics, topicDocs, topicTime, colorByTid };
}

async function setRun(K, seed) {
  App.K = K; App.seed = seed; updateSegs();
  await loadRun(K, seed);
  renderTopicList();
  let next = null;
  if (App.selRef != null) next = [...curTopics()].filter(t => t.ref === App.selRef).sort((a, b) => a.order - b.order)[0];
  selectTopic((next || curTopics()[0]).id, true);
  if (App.tab === "time") renderTime();
}

function selectTab(tab) {
  App.tab = tab;
  $$("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".view").forEach(v => v.classList.remove("active"));
  $("#view-" + tab).classList.add("active");
  $("#seed-seg").classList.toggle("disabled", tab === "stability");  // seed N/A on stability (all runs)
  if (tab === "time") renderTime();
  if (tab === "stability") renderGraph();
}

/* ---------- topic list ---------- */
function sparkPath(row, w = 56, h = 16) {
  const mx = Math.max(...row, 1e-9), step = w / (row.length - 1 || 1);
  return row.map((v, i) => `${i ? "L" : "M"}${(i * step).toFixed(1)} ${(h - (v / mx) * h).toFixed(1)}`).join(" ");
}
const topWords = (t, n) => [...t.words].sort((a, b) => b.p - a.p).slice(0, n).map(x => x.w);

function renderTopicList() {
  const ul = $("#topic-list"); ul.innerHTML = "";
  const time = curRun().topicTime, color = curRun().colorByTid;
  const maxPrev = Math.max(...curTopics().map(t => t.prevalence));
  const cmp = {
    order: (a, b) => a.order - b.order,
    prevalence: (a, b) => b.prevalence - a.prevalence,
    stability: (a, b) => b.stability - a.stability,
  }[App.sortMode];
  $("#topic-count").textContent = `${curTopics().length} topics`;
  for (const t of [...curTopics()].sort(cmp)) {
    const li = document.createElement("li");
    li.className = "topic-row" + (t.id === App.sel ? " sel" : "");
    li.onclick = () => selectTopic(t.id);
    const ws = topWords(t, 4);
    li.innerHTML = `
      <span class="topic-swatch" style="background:${t.color}"></span>
      <span class="words"><b>${ws[0]} ${ws[1]}</b> ${ws.slice(2).join(" ")}</span>
      <span class="meta">
        <svg class="spark" viewBox="0 0 56 16"><path d="${sparkPath(time.matrix[t.id])}"
             fill="none" stroke="${t.color}" stroke-width="1.5"/></svg>
        <span class="prev-bar"><i style="width:${100 * t.prevalence / maxPrev}%;background:${t.color}"></i></span>
        <span class="stab-dot">stab ${t.stability.toFixed(2)}</span>
      </span>`;
    ul.appendChild(li);
  }
}

/* ---------- topic detail ---------- */
function selectTopic(id, keepTau) {
  App.sel = id;
  const t = topicById(id);
  App.selRef = t.ref;
  renderTopicList();
  const docs = curRun().topicDocs[id].docs;
  if (!keepTau || App.tau == null) App.tau = docs[Math.min(200, docs.length - 1)][1];
  renderDetail(t);
}

function renderDetail(t) {
  const d = $("#topic-detail");
  const ws = topWords(t, 5).join(" · ");
  d.innerHTML = `
    <div class="detail-head">
      <span class="topic-swatch" style="width:14px;height:14px;border-radius:4px;background:${t.color}"></span>
      <h2>Topic ${t.id}</h2>
      <span class="pill">prevalence ${pct(t.prevalence)}</span>
      <span class="pill">stability ${t.stability.toFixed(2)}</span>
      <span class="pill">aligns → ref ${t.ref}</span>
    </div>
    <div class="muted">${ws}</div>
    <div class="detail-grid">
      <div class="card" id="words-card">
        <h3>Top words <span class="rhs">relevance λ</span></h3>
        <div id="wordbars"></div>
        <div class="lambda-row"><span>distinctive</span>
          <input type="range" id="lambda" min="0" max="1" step="0.05" value="${App.lambda}">
          <span>frequent</span></div>
      </div>
      <div class="card">
        <h3>Documents by P(topic | doc) <span class="rhs">drag the line ↕</span></h3>
        <div class="cdf-wrap" id="cdf"></div>
        <div class="cdf-hint" id="cdf-hint"></div>
      </div>
    </div>
    <div class="docs" id="docs"></div>`;
  $("#lambda").oninput = e => { App.lambda = +e.target.value; renderWords(t); };
  renderWords(t); renderThreshold(t); renderDocs(t);
}

function renderWords(t) {
  const box = $("#wordbars"); box.innerHTML = "";
  const scored = t.words.map(x => ({ ...x, s: App.lambda * Math.log(x.p) + (1 - App.lambda) * Math.log(x.r) }))
    .sort((a, b) => b.s - a.s).slice(0, 18);
  const maxP = Math.max(...scored.map(x => x.p));
  for (const x of scored) {
    const row = document.createElement("div");
    row.className = "wordbar";
    row.innerHTML = `<span class="w">${x.w}</span>
      <span class="track"><i style="width:${100 * x.p / maxP}%;background:${hexRgba(t.color, .85)}"></i></span>`;
    box.appendChild(row);
  }
}

const FLOOR = 0.05;  // truncate the long tail of near-zero P(topic|doc)
function renderThreshold(t) {
  const data = curRun().topicDocs[t.id];
  const host = $("#cdf");
  const pts = data.docs.filter(d => d[1] >= FLOOR);
  const W = host.clientWidth || 360, H = 158, m = { t: 10, r: 12, b: 30, l: 44 };
  const n = Math.max(pts.length, 1);
  const x = d3.scaleLinear().domain([0, n]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([FLOOR, Math.max(data.max, FLOOR + 1e-3)]).range([H - m.b, m.t]);
  App.tau = Math.max(FLOOR, Math.min(data.max, App.tau));

  const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const retain = svg.append("rect").attr("class", "cdf-sel").attr("x", m.l).attr("y", m.t).attr("height", H - m.b - m.t);
  svg.append("path").attr("class", "cdf-area")
    .attr("d", d3.area().x((d, i) => x(i + 1)).y0(H - m.b).y1(d => y(d[1]))(pts));
  svg.append("path").attr("class", "cdf-line")
    .attr("d", d3.line().x((d, i) => x(i + 1)).y(d => y(d[1]))(pts));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format("~s")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
    .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".2f")));
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 4).attr("text-anchor", "middle")
    .attr("fill", "var(--muted)").attr("font-size", 10).text("documents ranked by P(topic | doc)  →");

  const thLine = svg.append("line").attr("class", "cdf-line").attr("x1", m.l).attr("x2", W - m.r).attr("stroke-dasharray", "4 3");
  const handle = svg.append("circle").attr("class", "cdf-handle").attr("r", 7).attr("cx", W - m.r);
  const place = () => {
    const py = y(App.tau);
    thLine.attr("y1", py).attr("y2", py); handle.attr("cy", py);
    let c = 0; for (const d of pts) if (d[1] >= App.tau) c++;
    retain.attr("width", Math.max(0, x(c) - m.l));
    const more = c >= data.docs.length ? "+" : "";  // retained set hit the stored cap
    $("#cdf-hint").textContent =
      `P(topic|doc) ≥ ${App.tau.toFixed(3)} → ${c.toLocaleString()}${more} documents retained · tail truncated below ${FLOOR}`;
  };
  const setFromY = py => { App.tau = Math.max(FLOOR, Math.min(data.max, y.invert(py))); place(); renderDocs(t); };
  handle.call(d3.drag().on("drag", ev => setFromY(ev.y)));
  svg.on("click", ev => setFromY(d3.pointer(ev)[1]));
  host.innerHTML = ""; host.append(svg.node());
  place();
}

/* ---------- stratified doc sample ---------- */
function stratifiedSample(docs, tau, max, bins = 8, perBin = 3) {
  const sel = docs.filter(d => d[1] >= tau);
  if (sel.length <= bins * perBin) return sel;
  const span = Math.max(max - tau, 1e-9);
  const buckets = Array.from({ length: bins }, () => []);
  for (const d of sel) {
    let b = Math.floor((d[1] - tau) / span * bins);
    if (b >= bins) b = bins - 1; if (b < 0) b = 0;
    buckets[b].push(d);
  }
  const out = [];
  for (const bk of buckets) {
    if (!bk.length) continue;
    const stepi = bk.length / Math.min(perBin, bk.length);
    for (let i = 0; i < bk.length && out.length < bins * perBin + 4; i += stepi) out.push(bk[Math.floor(i)]);
  }
  return out.sort((a, b) => b[1] - a[1]);
}

function renderDocs(t) {
  const box = $("#docs"); if (!box) return;
  const data = curRun().topicDocs[t.id];
  const sample = stratifiedSample(data.docs, App.tau, data.max);
  box.innerHTML = `<div class="docs-head"><h3>Documents</h3>
    <span class="strat-note">${sample.length} shown · stratified high→low across P(topic|doc)</span></div>`;
  for (const [docIdx, p] of sample) {
    const meta = App.docsMeta[docIdx];
    const el = document.createElement("details");
    el.className = "doc";
    el.innerHTML = `<summary>
        <span class="pp">${p.toFixed(2)}</span>
        <span><span class="ti">${esc(meta.title) || "(untitled)"}</span>
        <span class="by">${meta.venue} · ${meta.year} · ${meta.id}</span></span>
      </summary>`;
    el.addEventListener("toggle", () => {
      if (el.open && !el.dataset.loaded) { el.dataset.loaded = "1"; loadDocBody(docIdx, el, t.id); }
    });
    box.appendChild(el);
  }
}

async function loadDocBody(docIdx, el, curTid) {
  const ss = App.manifest.shard_size, si = Math.floor(docIdx / ss), li = docIdx - si * ss;
  const [docShard, tokBuf, colBuf] = await Promise.all([
    getJSON(`${DATA}/docs/shard_${si}.json`),
    getBin(`${DATA}/docs/tokens_${si}.bin`),
    getBin(`${DATA}/${runKey()}/color_${si}.bin`),
  ]);
  const lengths = docShard.lengths;
  let off = 0; for (let i = 0; i < li; i++) off += lengths[i];
  const n = lengths[li];
  const ids = new Uint16Array(tokBuf, off * 2, n);
  const col = new Uint8Array(colBuf, off * 2, n * 2);
  const { html, present } = colorize(docShard.abstract[li] || "", ids, col, curTid);
  const body = document.createElement("div");
  body.className = "abs";
  body.innerHTML = html + legend(present);
  el.appendChild(body);
}

/* greedy align raw abstract words → processed tokens, color by dominant topic */
function colorize(abstract, ids, col, curTid) {
  const colors = curRun().colorByTid, vocab = App.vocab;
  const flat = [];
  for (let i = 0; i < ids.length; i++) {
    const surface = vocab[ids[i]] || "", topic = col[2 * i], prob = col[2 * i + 1];
    for (const part of surface.split("_")) flat.push({ w: part, topic, prob });
  }
  const present = new Set();
  let ptr = 0, html = "";
  for (const piece of abstract.split(/(\s+)/)) {
    if (/^\s+$/.test(piece) || piece === "") { html += piece; continue; }
    const mm = piece.match(/^([^\w]*)(.*?)([^\w]*)$/);
    const pre = mm[1], core = mm[2], post = mm[3], clean = core.toLowerCase();
    let matched = false;
    if (ptr < flat.length && clean) {
      const f = flat[ptr];
      if (f.w && (clean === f.w || clean.startsWith(f.w) || f.w.startsWith(clean))) {
        matched = true;
        if (f.topic !== 255) {
          present.add(f.topic);
          const cur = f.topic === curTid ? " cur" : "";
          const a = 0.14 + 0.55 * (f.prob / 100);
          html += `${esc(pre)}<span class="tok${cur}" style="background:${hexRgba(colors[f.topic] || "#888", a)};color:${colors[f.topic] || "#333"}">${esc(core)}</span>${esc(post)}`;
        } else html += esc(piece);
        ptr++;
      }
    }
    if (!matched) html += esc(piece);
  }
  return { html, present };
}
function legend(present) {
  if (!present.size) return "";
  const tops = curRun().topics.topics, colors = curRun().colorByTid;
  const items = [...present].slice(0, 14).map(tid => {
    const t = tops.find(x => x.id === tid);
    return `<span><i style="background:${colors[tid]}"></i>${t ? topWords(t, 2).join(" ") : tid}</span>`;
  }).join("");
  return `<div class="legend-mini">${items}</div>`;
}

/* ---------- time tab ---------- */
function yearCounts() {
  if (App._yc) return App._yc;
  const c = {}; for (const md of App.docsMeta) c[md.year] = (c[md.year] || 0) + 1;
  App._yc = c; return c;
}
function renderTime() {
  const host = $("#time-chart"); host.innerHTML = "";
  const T = curRun().topicTime, color = curRun().colorByTid, counts = yearCounts();
  // trim leading sparse years (adaptive: a fraction of the peak year's count, so it
  // works for both the 70k corpus and the ~1k subset); keep through the last year.
  const peak = Math.max(...T.years.map(y => counts[y] || 0));
  const thresh = Math.max(15, 0.1 * peak);
  let start = 0;
  while (start < T.years.length - 1 && (counts[T.years[start]] || 0) < thresh) start++;
  const years = T.years.slice(start), mat = T.matrix.map(row => row.slice(start)), hidden = start;

  const W = host.clientWidth || 900, H = host.clientHeight || 460, m = { t: 16, r: 134, b: 32, l: 46 };
  const x = d3.scaleLinear().domain([years[0], years[years.length - 1]]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, d3.max(mat, row => d3.max(row))]).nice().range([H - m.b, m.t]);
  const line = d3.line().x((_, i) => x(years[i])).y(v => y(v)).curve(d3.curveMonotoneX);
  const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`);

  const cmax = d3.max(years, yr => counts[yr]), cBandH = (H - m.b - m.t) * 0.30;
  const bw = Math.max(2, (W - m.r - m.l) / years.length * 0.55);
  svg.append("g").attr("opacity", 0.5).selectAll("rect").data(years).join("rect")
    .attr("x", yr => x(yr) - bw / 2).attr("width", bw)
    .attr("y", yr => (H - m.b) - cBandH * counts[yr] / cmax)
    .attr("height", yr => cBandH * counts[yr] / cmax).attr("fill", "#c7cedb");

  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`)
    .call(d3.axisBottom(x).ticks(Math.min(years.length, 10)).tickFormat(d3.format("d")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
    .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".1%")));
  svg.append("text").attr("x", m.l).attr("y", 11).attr("font-size", 10).attr("fill", "#7b8493")
    .text(hidden ? `grey bars = #docs/yr · ${hidden} earlier sparse year${hidden > 1 ? "s" : ""} hidden (< ${Math.round(thresh)} docs)` : "grey bars = #docs/yr");

  const g = svg.append("g");
  const order = mat.map((row, tid) => [tid, d3.max(row)]).sort((a, b) => a[1] - b[1]);
  for (const [tid] of order) {
    g.append("path").datum(mat[tid]).attr("class", "tline")
      .attr("stroke", color[tid]).attr("d", line).attr("data-tid", tid)
      .on("mousemove", ev => {
        showTip(`<b>Topic ${tid}</b> · peak ${pct(d3.max(mat[tid]))}<br><span class="tt-sub">${topWords(topicById(tid), 6).join(" ")}</span>`, ev);
        svg.selectAll(".tline").classed("dim", true).classed("hot", false);
        d3.select(ev.currentTarget).classed("dim", false).classed("hot", true).raise();
      })
      .on("mouseleave", () => { hideTip(); svg.selectAll(".tline").classed("dim", false).classed("hot", false); })
      .on("click", () => { selectTopic(tid); selectTab("topics"); });
  }
  for (const [tid] of order.slice(-8)) {
    const row = mat[tid];
    svg.append("text").attr("x", x(years[years.length - 1]) + 6).attr("y", y(row[row.length - 1]))
      .attr("dy", "0.32em").attr("font-size", 11).attr("fill", color[tid])
      .text(topWords(topicById(tid), 2).join(" "));
  }
  host.append(svg.node());
}

/* ---------- stability graph ---------- */
function bindStabilityControls() {
  const sync = () => {
    $("#stab-min-v").textContent = (+$("#stab-min").value).toFixed(2);
    $("#edge-min-v").textContent = (+$("#edge-min").value).toFixed(2);
  };
  ["stab-min", "edge-min"].forEach(id => $("#" + id).addEventListener("input", () => { sync(); debouncedGraph(); }));
  $("#stab-color").addEventListener("change", () => renderGraph());
  sync();
}
let graphTimer = null;
function debouncedGraph() { clearTimeout(graphTimer); graphTimer = setTimeout(renderGraph, 120); }

async function renderGraph() {
  const host = $("#graph");
  if (!App.graph) App.graph = await getJSON(`${DATA}/stability/graph.json`);
  if (App.sim) App.sim.stop();
  host.innerHTML = "";
  const W = host.clientWidth || 900, H = host.clientHeight || 520;
  const stabMin = +$("#stab-min").value, edgeMin = +$("#edge-min").value, colorMode = $("#stab-color").value;

  const nodes = App.graph.nodes.filter(n => n.stability >= stabMin).map(n => ({ ...n }));
  const ids = new Set(nodes.map(n => n.id));
  const links = App.graph.edges.filter(e => e.weight >= edgeMin && ids.has(e.source) && ids.has(e.target)).map(e => ({ ...e }));

  const kColor = d3.scaleOrdinal().domain(App.manifest.ks).range(["#3b6fd4", "#6a4fd0", "#c0497f", "#d98a3a"]);
  const stabColor = d3.scaleSequential(d3.interpolateViridis).domain([stabMin, 1]);
  const fill = n => colorMode === "k" ? kColor(n.k) : stabColor(n.stability);
  const rad = n => Math.max(2.5, Math.sqrt(n.prevalence) * 70);

  const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const root = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.3, 6]).on("zoom", ev => root.attr("transform", ev.transform)));

  const link = root.append("g").attr("stroke-opacity", 0.5).selectAll("line")
    .data(links).join("line").attr("class", "link").attr("stroke-width", d => 0.4 + 2.2 * d.weight);
  const node = root.append("g").selectAll("circle").data(nodes).join("circle")
    .attr("class", d => "node" + (d.seed === App.manifest.canonical_seed ? " canon" : ""))
    .attr("r", rad).attr("fill", fill)
    .on("mousemove", (ev, d) => showTip(
      `<b>${d.label}</b><br><span class="tt-sub">K=${d.k} · seed ${d.seed} · topic ${d.topic}<br>stability ${d.stability.toFixed(3)} · prev ${pct(d.prevalence)}</span>`, ev))
    .on("mouseleave", hideTip)
    .on("click", (ev, d) => openNode(d))
    .call(d3.drag()
      .on("start", (ev, d) => { if (!ev.active) App.sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end", (ev, d) => { if (!ev.active) App.sim.alphaTarget(0); d.fx = null; d.fy = null; }));

  App.sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(d => 10 + 46 * (1 - d.weight)).strength(d => d.weight))
    .force("charge", d3.forceManyBody().strength(-26))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide().radius(d => rad(d) + 1.5))
    .on("tick", () => {
      link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
    });
  drawLegend(svg, colorMode, stabMin, kColor, stabColor);
  host.append(svg.node());
}

async function openNode(d) {
  // every run is browseable now: jump straight to that K / seed / topic
  App.K = d.k; App.seed = d.seed; updateSegs();
  await loadRun(d.k, d.seed);
  renderTopicList();
  selectTopic(d.topic);
  selectTab("topics");
}

function drawLegend(svg, colorMode, stabMin, kColor, stabColor) {
  const INK = "#1c2230", MUTED = "#7b8493", LINE = "#d4d9e2";
  const g = svg.append("g").attr("transform", "translate(14,14)").attr("pointer-events", "none");
  const h = colorMode === "k" ? 150 : 126;
  g.append("rect").attr("width", 196).attr("height", h).attr("rx", 9).attr("fill", "rgba(255,255,255,0.93)").attr("stroke", LINE);
  g.append("text").attr("x", 12).attr("y", 20).attr("font-size", 11).attr("font-weight", 600).attr("fill", INK).text("size = topic prevalence");
  let cx = 30;
  for (const [p, lab] of [[0.005, "0.5%"], [0.03, "3%"]]) {
    const r = Math.max(2.5, Math.sqrt(p) * 70);
    g.append("circle").attr("cx", cx).attr("cy", 48).attr("r", r).attr("fill", "none").attr("stroke", "#9aa3b3");
    g.append("text").attr("x", cx).attr("y", 70).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", MUTED).text(lab);
    cx += 78;
  }
  g.append("text").attr("x", 12).attr("y", 92).attr("font-size", 11).attr("font-weight", 600).attr("fill", INK)
    .text(colorMode === "k" ? "color = K (num. topics)" : "color = stability score");
  if (colorMode === "k") {
    let lx = 12;
    for (const k of App.manifest.ks) {
      g.append("rect").attr("x", lx).attr("y", 104).attr("width", 12).attr("height", 12).attr("rx", 3).attr("fill", kColor(k));
      g.append("text").attr("x", lx + 16).attr("y", 114).attr("font-size", 10).attr("fill", MUTED).text(k);
      lx += 45;
    }
  } else {
    const defs = svg.append("defs");
    const lg = defs.append("linearGradient").attr("id", "stabgrad");
    for (let i = 0; i <= 10; i++) lg.append("stop").attr("offset", `${i * 10}%`).attr("stop-color", stabColor(stabMin + (1 - stabMin) * i / 10));
    g.append("rect").attr("x", 12).attr("y", 104).attr("width", 162).attr("height", 10).attr("rx", 2).attr("fill", "url(#stabgrad)");
    g.append("text").attr("x", 12).attr("y", 124).attr("font-size", 9.5).attr("fill", MUTED).text(stabMin.toFixed(2) + " (less stable)");
    g.append("text").attr("x", 174).attr("y", 124).attr("text-anchor", "end").attr("font-size", 9.5).attr("fill", MUTED).text("1.0");
  }
}

init().catch(e => { document.body.innerHTML = `<pre style="padding:24px;color:#b00">${e.stack || e}</pre>`; });
