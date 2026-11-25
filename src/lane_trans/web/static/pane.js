import { Graph } from "./graph.js";

// One view shared by the input and result panes, so they always show the same region.
export const view = { scale: 10, ox: 40, oy: 0, ready: false }; // px per metre; screen pos of world (0,0)

const C = { edge: "#4C72B0", mod: "#C44E52", ghost: "#b9b9b9", grid: "#ececec", axis: "#bdbdbd",
  hub: "#DD8452", end: "#55A868", mid: "#F2C14E", lone: "#9a9a9a" };

// Canvas that draws a graph and handles pan/zoom. Subclasses override grab/dragStart/dragMove/click.
export class Pane {
  constructor(canvas, { graph = new Graph(), ghost = () => null, onView = () => {}, onCursor = () => {} } = {}) {
    Object.assign(this, { canvas, graph, ghost, onView, onCursor });
    this.ctx = canvas.getContext("2d");
    this.cursor = null; // screen position of the pointer, null when outside
    this.gridStep = 1;
    this.g = null;      // gesture in progress
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.bindPointer();
  }

  sx(x) { return view.ox + x * view.scale; }
  sy(y) { return view.oy - y * view.scale; }
  wx(px) { return (px - view.ox) / view.scale; }
  wy(py) { return (view.oy - py) / view.scale; }

  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.canvas.getBoundingClientRect();
    this.w = r.width;
    this.h = r.height;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!view.ready && this.h > 0) { view.oy = this.h - 40; view.ready = true; }
    this.draw();
  }

  fit(box) {
    if (!box || !this.w) return;
    const m = 50, bw = Math.max(box.maxX - box.minX, 1), bh = Math.max(box.maxY - box.minY, 1);
    view.scale = Math.min((this.w - 2 * m) / bw, (this.h - 2 * m) / bh);
    view.ox = (this.w - bw * view.scale) / 2 - box.minX * view.scale;
    view.oy = (this.h + bh * view.scale) / 2 + box.minY * view.scale;
    this.onView();
  }

  // ---- gestures: left-drag on a grabbed object moves it, anything else pans; no drag = click
  grab(p) { return null; }
  dragStart(grab) {}
  dragMove(grab, p) {}
  click(p, button) {}

  bindPointer() {
    const c = this.canvas;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      const p = this.local(e);
      this.g = { button: e.button, p0: p, moved: false, ox: view.ox, oy: view.oy,
        grab: e.button === 0 ? this.grab(p) : null };
    });
    c.addEventListener("pointermove", (e) => {
      const p = this.local(e), g = this.g;
      this.cursor = p;
      this.onCursor(this.wx(p.x), this.wy(p.y));
      if (g && !g.moved && Math.hypot(p.x - g.p0.x, p.y - g.p0.y) > 4) {
        g.moved = true;
        if (g.grab) this.dragStart(g.grab);
      }
      if (g && g.moved && !g.grab) {
        view.ox = g.ox + p.x - g.p0.x;
        view.oy = g.oy + p.y - g.p0.y;
        return this.onView();
      }
      if (g && g.moved) this.dragMove(g.grab, p);
      this.draw();
    });
    c.addEventListener("pointerup", (e) => {
      const g = this.g;
      this.g = null;
      if (g && !g.moved) this.click(this.local(e), g.button);
    });
    c.addEventListener("pointerleave", () => { this.cursor = null; this.draw(); });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const p = this.local(e), wx = this.wx(p.x), wy = this.wy(p.y);
      view.scale = Math.min(Math.max(view.scale * Math.exp(-e.deltaY * 0.0015), 0.05), 5000);
      view.ox = p.x - wx * view.scale;
      view.oy = p.y + wy * view.scale;
      this.onView();
    }, { passive: false });
  }

  // ---- hit tests (screen space); return a node id / {i, x, y} on the edge, or null
  hitNode(p, r = 10) {
    let best = null, bd = r;
    for (const n of this.graph.nodes.values()) {
      const d = Math.hypot(this.sx(n.x) - p.x, this.sy(n.y) - p.y);
      if (d <= bd) { bd = d; best = n.id; }
    }
    return best;
  }

  hitEdge(p, r = 6) {
    let best = null, bd = r;
    this.graph.edges.forEach((e, i) => {
      const a = this.graph.nodes.get(e.a), b = this.graph.nodes.get(e.b);
      const ax = this.sx(a.x), ay = this.sy(a.y), dx = this.sx(b.x) - ax, dy = this.sy(b.y) - ay;
      const t = Math.min(1, Math.max(0, ((p.x - ax) * dx + (p.y - ay) * dy) / (dx * dx + dy * dy || 1)));
      const d = Math.hypot(ax + t * dx - p.x, ay + t * dy - p.y);
      if (d <= bd) { bd = d; best = { i, x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) }; }
    });
    return best;
  }

  // ---- drawing
  draw() {
    if (!this.w) return;
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.drawGrid();
    const ghost = this.ghost();
    if (ghost) this.strokeEdges(ghost, () => true, C.ghost, 2, [5, 4]);
    this.strokeEdges(this.graph, (e) => !e.mod, C.edge, 2);
    this.strokeEdges(this.graph, (e) => e.mod, C.mod, 3);
    this.drawNodes();
    this.drawOverlay();
  }

  drawOverlay() {}

  strokeEdges(graph, keep, color, width, dash = []) {
    const ctx = this.ctx;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.lineCap = "round";
    ctx.beginPath();
    for (const e of graph.edges) {
      if (!keep(e)) continue;
      const a = graph.nodes.get(e.a), b = graph.nodes.get(e.b);
      ctx.moveTo(this.sx(a.x), this.sy(a.y));
      ctx.lineTo(this.sx(b.x), this.sy(b.y));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawNodes() {
    const ctx = this.ctx, deg = this.graph.degrees();
    for (const n of this.graph.nodes.values()) {
      const d = deg.get(n.id);
      ctx.fillStyle = d >= 3 ? C.hub : d === 1 ? C.end : d === 2 ? C.mid : C.lone;
      ctx.beginPath();
      ctx.arc(this.sx(n.x), this.sy(n.y), d >= 3 ? 6 : d === 2 ? 3.5 : 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.lineWidth = 0.8; ctx.strokeStyle = "#222"; ctx.stroke();
    }
  }

  drawGrid() {
    const ctx = this.ctx, raw = 50 / view.scale, p10 = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * p10).find((v) => v >= raw);
    this.gridStep = step;
    ctx.lineWidth = 1; ctx.setLineDash([]); ctx.font = "10px sans-serif"; ctx.fillStyle = "#999";
    const label = (v) => String(Number(v.toPrecision(6)));
    for (let i = Math.ceil(this.wx(0) / step); i <= Math.floor(this.wx(this.w) / step); i++) {
      const x = Math.round(this.sx(i * step)) + 0.5;
      ctx.strokeStyle = i === 0 ? C.axis : C.grid;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.h); ctx.stroke();
      ctx.textBaseline = "bottom"; ctx.fillText(label(i * step), x + 2, this.h - 2);
    }
    for (let i = Math.ceil(this.wy(this.h) / step); i <= Math.floor(this.wy(0) / step); i++) {
      const y = Math.round(this.sy(i * step)) + 0.5;
      ctx.strokeStyle = i === 0 ? C.axis : C.grid;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.w, y); ctx.stroke();
      ctx.textBaseline = "top"; ctx.fillText(label(i * step), 2, y + 2);
    }
  }
}
