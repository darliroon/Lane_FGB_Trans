import { Graph } from "./graph.js";
import { Pane } from "./pane.js";

// Editable input pane. Tools: "draw" (click to add nodes / link / split edges), "move", "erase".
export class EditorPane extends Pane {
  constructor(canvas, opts = {}) {
    super(canvas, opts);
    this.onChange = opts.onChange || (() => {});
    this.tool = "draw";
    this.snap = true;
    this.active = null;   // node the next click in draw mode connects from
    this.history = [];
    this.future = [];
    this.setTool("draw");
  }

  setTool(t) {
    this.tool = t;
    this.active = null;
    this.canvas.style.cursor = { draw: "crosshair", move: "default", erase: "not-allowed" }[t];
    this.draw();
  }

  // world position of the pointer, snapped to the visible grid when enabled
  world(p) {
    let x = this.wx(p.x), y = this.wy(p.y);
    if (this.snap) {
      const s = this.gridStep;
      x = Math.round(x / s) * s;
      y = Math.round(y / s) * s;
    }
    return { x: +x.toFixed(6), y: +y.toFixed(6) };
  }

  target(p) {
    const node = this.hitNode(p);
    return { node, edge: node === null ? this.hitEdge(p) : null };
  }

  // ---- history (snapshot before every mutation)
  snapshot() {
    this.history.push(JSON.stringify(this.graph.toJSON()));
    if (this.history.length > 200) this.history.shift();
    this.future = [];
  }

  restore(json) {
    this.graph = Graph.fromJSON(JSON.parse(json));
    this.active = null;
    this.changed();
  }

  undo() {
    if (!this.history.length) return;
    this.future.push(JSON.stringify(this.graph.toJSON()));
    this.restore(this.history.pop());
  }

  redo() {
    if (!this.future.length) return;
    this.history.push(JSON.stringify(this.graph.toJSON()));
    this.restore(this.future.pop());
  }

  changed() {
    this.onChange();
    this.draw();
  }

  // replace the whole graph (import / example / clear); optimizer marks never belong to the input
  setGraph(g) {
    this.snapshot();
    g.edges.forEach((e) => { e.mod = false; });
    this.graph = g;
    this.active = null;
    this.changed();
  }

  cancel() {
    this.active = null;
    this.draw();
  }

  // ---- gestures
  grab(p) {
    if (this.tool === "erase") return null;
    const node = this.hitNode(p);
    return node === null ? null : { node };
  }

  dragStart() { this.snapshot(); }

  dragMove(grab, p) {
    const n = this.graph.nodes.get(grab.node), w = this.world(p);
    n.x = w.x;
    n.y = w.y;
    this.onChange();
  }

  click(p, button) {
    if (button === 2) {                 // right click: stop the chain, or erase when idle
      if (this.active !== null) return this.cancel();
      return this.erase(p);
    }
    if (button !== 0) return;
    if (this.tool === "erase") return this.erase(p);
    if (this.tool === "draw") this.place(p);
  }

  erase(p) {
    const { node, edge } = this.target(p);
    if (node === null && !edge) return;
    this.snapshot();
    if (node !== null) this.graph.removeNode(node);
    else this.graph.removeEdge(edge.i);
    this.active = null;
    this.changed();
  }

  place(p) {
    const g = this.graph, { node, edge } = this.target(p);
    let id;
    if (node !== null) {
      id = node;
      if (this.active !== null && id !== this.active && !g.hasEdge(this.active, id)) {
        this.snapshot();
        g.addEdge(this.active, id);
      }
    } else if (edge) {                  // click on an edge: split it with a new node
      this.snapshot();
      const { a, b, mod } = g.edges[edge.i];
      g.removeEdge(edge.i);
      id = g.addNode(edge.x, edge.y);
      g.addEdge(a, id, mod);
      g.addEdge(id, b, mod);
      if (this.active !== null) g.addEdge(this.active, id);
    } else {
      this.snapshot();
      const w = this.world(p);
      id = g.addNode(w.x, w.y);
      if (this.active !== null) g.addEdge(this.active, id);
    }
    this.active = id;
    this.changed();
  }

  // ---- overlay: rubber band, hover feedback
  drawOverlay() {
    const ctx = this.ctx, g = this.graph;
    const ring = (x, y, color, r = 9) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(this.sx(x), this.sy(y), r, 0, 2 * Math.PI); ctx.stroke();
    };
    const act = this.active !== null ? g.nodes.get(this.active) : null;
    if (act) ring(act.x, act.y, "#2c7be5", 10);
    if (!this.cursor || (this.g && this.g.moved)) return;

    const { node, edge } = this.target(this.cursor);
    if (this.tool === "erase") {
      if (node !== null) ring(g.nodes.get(node).x, g.nodes.get(node).y, "#d33", 10);
      else if (edge) ring(edge.x, edge.y, "#d33", 6);
      return;
    }
    if (this.tool === "move") {
      if (node !== null) ring(g.nodes.get(node).x, g.nodes.get(node).y, "#2c7be5", 10);
      return;
    }
    let end;                            // draw tool
    if (node !== null) { end = g.nodes.get(node); ring(end.x, end.y, "#2c7be5", 10); }
    else if (edge) { end = edge; ring(edge.x, edge.y, "#2c7be5", 6); }
    else end = this.world(this.cursor);
    if (act) {
      ctx.strokeStyle = "#2c7be5"; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(this.sx(act.x), this.sy(act.y)); ctx.lineTo(this.sx(end.x), this.sy(end.y));
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (node === null && !edge) {
      ctx.fillStyle = "rgba(44,123,229,.45)";
      ctx.beginPath(); ctx.arc(this.sx(end.x), this.sy(end.y), 4, 0, 2 * Math.PI); ctx.fill();
    }
  }
}
