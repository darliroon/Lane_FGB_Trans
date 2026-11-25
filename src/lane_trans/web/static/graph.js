// Undirected graph in local metres. Ids are internal; toJSON() re-indexes nodes to 0..n-1.
export class Graph {
  constructor() {
    this.nodes = new Map(); // id -> {id, x, y}
    this.edges = [];        // {a, b, mod}; mod = chain was rewritten by the optimizer
    this.nextId = 0;
  }

  addNode(x, y) {
    const id = this.nextId++;
    this.nodes.set(id, { id, x, y });
    return id;
  }

  hasEdge(a, b) {
    return this.edges.some((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a));
  }

  addEdge(a, b, mod = false) {
    if (a === b || this.hasEdge(a, b)) return false;
    this.edges.push({ a, b, mod });
    return true;
  }

  removeNode(id) {
    this.nodes.delete(id);
    this.edges = this.edges.filter((e) => e.a !== id && e.b !== id);
  }

  removeEdge(i) {
    this.edges.splice(i, 1);
  }

  degrees() {
    const d = new Map([...this.nodes.keys()].map((id) => [id, 0]));
    for (const e of this.edges) {
      d.set(e.a, d.get(e.a) + 1);
      d.set(e.b, d.get(e.b) + 1);
    }
    return d;
  }

  bbox() {
    if (!this.nodes.size) return null;
    const b = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
    for (const { x, y } of this.nodes.values()) {
      b.minX = Math.min(b.minX, x); b.maxX = Math.max(b.maxX, x);
      b.minY = Math.min(b.minY, y); b.maxY = Math.max(b.maxY, y);
    }
    return b;
  }

  toJSON() {
    const idx = new Map([...this.nodes.keys()].map((id, i) => [id, i]));
    return {
      nodes: [...this.nodes.values()].map((n) => ({ id: idx.get(n.id), x: n.x, y: n.y })),
      edges: this.edges.map((e) => ({ source: idx.get(e.a), target: idx.get(e.b), mod: e.mod })),
    };
  }

  static fromJSON(d) {
    if (!d || !Array.isArray(d.nodes) || !Array.isArray(d.edges)) {
      throw new Error("需要包含 nodes 和 edges 数组的 JSON");
    }
    const g = new Graph();
    const map = new Map();
    for (const n of d.nodes) {
      const x = Number(n.x), y = Number(n.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error(`节点 ${n.id} 的坐标无效`);
      map.set(n.id, g.addNode(x, y));
    }
    for (const e of d.edges) {
      const a = map.get(e.source), b = map.get(e.target);
      if (a === undefined || b === undefined) throw new Error(`边 ${e.source}-${e.target} 引用了不存在的节点`);
      g.addEdge(a, b, !!e.mod);
    }
    return g;
  }
}
