import { Graph } from "./graph.js";
import { Pane } from "./pane.js";
import { EditorPane } from "./editor.js";

const $ = (id) => document.getElementById(id);
const PARAMS = ["max_loop_len", "connector_margin", "parallel_ratio"];
let defaults = {}, result = null, resultLog = [], busy = false;

const redrawBoth = () => { editor.draw(); output.draw(); };
const showCursor = (x, y) => { $("cursorPos").textContent = `x = ${x.toFixed(2)} m,  y = ${y.toFixed(2)} m`; };

const editor = new EditorPane($("input"), { onView: redrawBoth, onCursor: showCursor, onChange: onInputChange });
const output = new Pane($("result"), {
  onView: redrawBoth, onCursor: showCursor,
  ghost: () => ($("ghost").checked && result ? editor.graph : null),
});

function counts(g) {
  return `${g.nodes.size} 节点 / ${g.edges.length} 边`;
}

function setMsg(text, err = false) {
  $("msg").textContent = text;
  $("msg").className = err ? "err" : "";
}

function onInputChange() {
  $("inMeta").textContent = counts(editor.graph);
  $("undo").disabled = !editor.history.length;
  $("redo").disabled = !editor.future.length;
  $("stale").style.display = result ? "block" : "none";
  output.draw();
}

function showResult(graph, log, stats) {
  result = graph;
  resultLog = log;
  output.graph = graph;
  $("placeholder").style.display = "none";
  $("stale").style.display = "none";
  $("outMeta").textContent = counts(graph);
  $("exportResult").disabled = false;

  const rows = [["节点", "nodes"], ["边", "edges"], ["环 (cycle basis)", "loops"], ["连通块", "components"], ["悬挂端点", "dangling_ends"]];
  $("stats").innerHTML = "<tr><th></th><th>转换前</th><th>转换后</th></tr>" + rows
    .map(([label, k]) => `<tr><td>${label}</td><td>${stats.before[k]}</td><td>${stats.after[k]}</td></tr>`).join("");

  const ol = $("log");
  ol.innerHTML = "";
  if (!log.length) ol.innerHTML = '<li style="color:#999;list-style:none;margin-left:-20px">没有需要处理的小环</li>';
  for (const line of log) {
    const li = document.createElement("li");
    li.textContent = line.replace("lanes=", "边序号=");  // remove_loops calls input edges "lanes"
    ol.appendChild(li);
  }
  output.draw();
}

async function convert() {
  if (busy) return;
  if (!editor.graph.edges.length) return setMsg("请先画至少一条边", true);
  busy = true;
  $("convert").disabled = true;
  $("convert").textContent = "转换中…";
  setMsg("");
  try {
    const params = {};
    for (const k of PARAMS) params[k] = $("p_" + k).value === "" ? undefined : Number($("p_" + k).value);
    const res = await fetch("/api/convert", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...editor.graph.toJSON(), params }),
    });
    const body = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
    showResult(Graph.fromJSON(body.graph), body.log, body.stats);
    setMsg(body.log.length ? `完成，共 ${body.log.length} 步` : "完成，图中没有需要处理的小环");
  } catch (e) {
    setMsg("转换失败：" + e.message, true);
  } finally {
    busy = false;
    $("convert").disabled = false;
    $("convert").textContent = "转换 ▶";
  }
}

function download(name, obj) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 1)], { type: "application/json" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function setTool(t) {
  editor.setTool(t);
  document.querySelectorAll("#tools button").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
}

async function loadExample() {
  const res = await fetch("/api/example");
  if (!res.ok) throw new Error("服务器上没有示例数据");
  const g = Graph.fromJSON(await res.json());
  editor.setGraph(g);
  editor.fit(g.bbox());
}

function guard(fn) {
  return async (...a) => { try { await fn(...a); } catch (e) { setMsg(e.message, true); } };
}

// ---- wiring
document.querySelectorAll("#tools button").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
$("convert").addEventListener("click", convert);
$("undo").addEventListener("click", () => editor.undo());
$("redo").addEventListener("click", () => editor.redo());
$("clear").addEventListener("click", () => { if (editor.graph.nodes.size) editor.setGraph(new Graph()); });
$("snap").addEventListener("change", (e) => { editor.snap = e.target.checked; });
$("ghost").addEventListener("change", () => output.draw());
$("fit").addEventListener("click", () => {
  editor.fit(editor.graph.bbox() || output.graph.bbox());
});
$("example").addEventListener("click", guard(loadExample));
$("export").addEventListener("click", () => download("input_graph.json", editor.graph.toJSON()));
$("exportResult").addEventListener("click", () => result && download("result_graph.json", { ...result.toJSON(), log: resultLog }));
$("import").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", guard(async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  const g = Graph.fromJSON(JSON.parse(await f.text()));
  editor.setGraph(g);
  editor.fit(g.bbox());
}));
$("resetParams").addEventListener("click", () => PARAMS.forEach((k) => { $("p_" + k).value = defaults[k]; }));

window.addEventListener("keydown", (e) => {
  if (e.target.matches("input")) return;
  const mod = e.ctrlKey || e.metaKey, k = e.key.toLowerCase();
  if (mod && k === "z") { e.preventDefault(); e.shiftKey ? editor.redo() : editor.undo(); }
  else if (mod && k === "y") { e.preventDefault(); editor.redo(); }
  else if (e.key === "Escape") editor.cancel();
  else if (!mod && k === "1") setTool("draw");
  else if (!mod && k === "2") setTool("move");
  else if (!mod && k === "3") setTool("erase");
});

(async () => {
  try {
    const cfg = await (await fetch("/api/config")).json();
    defaults = cfg.defaults;
    PARAMS.forEach((k) => { $("p_" + k).value = defaults[k]; });
    $("example").disabled = !cfg.example;
  } catch { setMsg("无法连接服务器，请确认 app.py 正在运行", true); }
  onInputChange();
  redrawBoth();
})();
