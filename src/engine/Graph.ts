// src/engine/Graph.ts
import { GState } from '../types'

export type Pt = { x: number, y: number }
type Edge = { u: number; v: number; visible: boolean }
type State = {
  vertices: { index: number, x: number, y: number, color: number, visible: boolean }[]
  edges: Edge[]
  periphery: number[]
  meta: { total_vertices: number, edges: number, periphery_size: number }
  labelMode: number
}

class WorkerRPC {
  private w: Worker
  private rid = 1
  private wait = new Map<number, (d: unknown, ok: boolean)=>void>()
  constructor() {
    this.w = new Worker('/py-worker.js')
    this.w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const { rid, ok, data, error } = e.data || {}
      const cb = this.wait.get(rid)
      if (cb) { this.wait.delete(rid); cb(ok ? data : error, !!ok) }
    }
  }
  call(cmd: string, payload?: unknown): Promise<unknown> {
    const id = this.rid++
    return new Promise((resolve, reject) => {
      this.wait.set(id, (d, ok) => ok ? resolve(d) : reject(d))
      this.w.postMessage({ cmd, payload, rid: id } as WorkerMessage)
    })
  }
}

class Vertex {
  private idx: number
  private pos: Pt
  private colorIndex = 1
  private visible = true
  private diameter = 30
  constructor(idx: number, pos: Pt, colorIndex: number) {
    this.idx = idx; this.pos = { ...pos }; this.colorIndex = colorIndex
    this.diameter = Vertex.calcDiameter(idx)
  }
  static calcDiameter(idx: number): number {
    if (idx >= 1000) return 40
    if (idx >= 100) return 36
    if (idx >= 10) return 32
    return 30
  }
  getIndex() { return this.idx }
  getPosition() { return this.pos }
  setPosition(p: Pt) { this.pos = { x: p.x, y: p.y } }
  isVisible() { return this.visible }
  setVisible(v: boolean) { this.visible = !!v }
  getDiameter() { return this.diameter }
  setDiameter(d: number) { this.diameter = d }
  getColorIndex() { return this.colorIndex }
  setColorIndex(c: number) { this.colorIndex = c }
}

export class Graph {
  vertices: Vertex[] = []
  edges: Edge[] = []
  periphery = {
    getIndices: () => this._periphery.slice(),
    neighborsOnPeriphery: (u: number): [number|null, number|null] => {
      const i = this._periphery.indexOf(u)
      if (i < 0 || this._periphery.length === 0) return [null, null]
      const n = this._periphery.length
      return [this._periphery[(i - 1 + n) % n] ?? null, this._periphery[(i + 1) % n] ?? null]
    }
  }
  onLayoutChanged: null | (() => void) = null
  lastAddInfo: { index: number, spawn_pos?: Pt, final_pos?: Pt } | null = null

  private rpc = new WorkerRPC()
  private _periphery: number[] = []
  private _labelMode = 2
  private _meta = { total_vertices: 0, edges: 0, periphery_size: 0 }

  // --- public API expected by CanvasGraph ---
  get_label_mode() { return this._labelMode }
  setLabelMode(mode: number) {
    this._labelMode = Math.max(0, Math.min(2, Math.floor(mode)))
    this.rpc.call('set_label_mode', { mode: this._labelMode }).catch(()=>{})
  }

  async startBasicGraph(n: number) {
    const raw = await this.rpc.call('start', { n: Math.max(3, Math.min(10, Math.floor(n))) })
    this._applyState(JSON.parse(raw) as State)
  }

  async addRandomVertex(): Promise<[boolean, number]> {
    const raw = await this.rpc.call('add_random')
    const res = JSON.parse(raw)
    this.lastAddInfo = res.info || null
    this._applyState(res.state as State)
    return [!!res.ok, Number(res.index)]
  }

  async addVertexBySelection(a: number, b: number): Promise<[boolean, number]> {
    const raw = await this.rpc.call('add_by_selection', { a, b })
    const res = JSON.parse(raw)
    this.lastAddInfo = res.info || null
    this._applyState(res.state as State)
    return [!!res.ok, Number(res.index)]
  }

  async reembedTutte() {
    const raw = await this.rpc.call('redraw')
    this._applyState(JSON.parse(raw as string) as State)
  }

  async declutterView(intensity = 1.0): Promise<boolean> {
    const raw = await this.rpc.call('declutter', { intensity })
    const res = JSON.parse(raw)
    this._applyState(res.state as State)
    return !!res.changed
  }

  async setAutoTutte(on: boolean) { await this.rpc.call('set_auto_tutte', { on: !!on }) }
  async setFinalizeMode(on: boolean) { await this.rpc.call('set_finalize', { on: !!on }) }

  async goToVertex(m: number) {
    const raw = await this.rpc.call('go_to', { m: Math.max(1, Math.floor(m)) })
    this._applyState(JSON.parse(raw) as State)
  }

  async saveToJsonObject(): Promise<GState> {
    const raw = await this.rpc.call('save')
    return JSON.parse(raw as string) as GState
  }

  async loadFromJsonObject(obj: GState) {
    const raw = await this.rpc.call('load', { json: JSON.stringify(obj) })
    this._applyState(JSON.parse(raw as string) as State)
  }

  get_center(): Pt {
    const per = this._periphery
    if (per.length < 1) return { x: 0, y: 0 }
    let sx = 0, sy = 0
    for (const i of per) { const p = this.vertices[i].getPosition(); sx += p.x; sy += p.y }
    return { x: sx / per.length, y: sy / per.length }
  }

  get_bounding_box(): [number, number, number, number] {
    const xs: number[] = [], ys: number[] = []
    for (const v of this.vertices) if (v && v.isVisible()) { const p = v.getPosition(); xs.push(p.x); ys.push(p.y) }
    if (!xs.length) return [0,0,0,0]
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
  }

  getStats() { return { ...this._meta } }

  // --- private ---
  private _applyState(s: State) {
    // vertices: build index-addressable array so graph.vertices[idx] is that vertex
    let maxIdx = -1
    for (const v of s.vertices) { if (v.index > maxIdx) maxIdx = v.index }
    const arr: Array<Vertex | undefined> = new Array(Math.max(0, maxIdx + 1))
    for (const v of s.vertices) {
      const vv = new Vertex(v.index, { x: v.x, y: v.y }, v.color)
      vv.setVisible(!!v.visible)
      arr[v.index] = vv
    }
    // Type says Vertex[], but we intentionally allow sparse; casting is safe for read sites with guards
    this.vertices = arr as unknown as Vertex[]
    // edges/periphery/meta
    this.edges = s.edges || []
    this._periphery = s.periphery || []
    this._meta = s.meta || this._meta
    this._labelMode = typeof s.labelMode === 'number' ? s.labelMode : this._labelMode
    // notify
    if (this.onLayoutChanged) this.onLayoutChanged()
  }
}