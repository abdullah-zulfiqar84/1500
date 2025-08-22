// src/engine/Graph.ts
import { QuadTree, type Point } from './QuadTree'

export type Pt = { x: number, y: number }
type Edge = { u: number; v: number; visible: boolean }
type State = {
  vertices: { index: number, x: number, y: number, color: number, visible: boolean }[]
  edges: Edge[]
  periphery: number[]
  meta: { total_vertices: number, edges: number, periphery_size: number }
  labelMode: number
}

// Advanced layout configuration
export interface AdvancedLayoutConfig {
  // Force-directed parameters
  springK: number
  repulsionStrength: number
  repulsionCutoff: number
  
  // Congestion parameters
  congestionThreshold: number
  congestionBeta: number
  
  // Performance parameters
  useSpatialIndexing: boolean
  maxIterations: number
  convergenceThreshold: number
  
  // Advanced features
  magneticForces: boolean
  adaptiveForces: boolean
  edgeBundling: boolean
}

class WorkerRPC {
  private w: Worker
  private rid = 1
  private wait = new Map<number, (d: any, ok: boolean)=>void>()
  constructor() {
    this.w = new Worker('/py-worker.js')
    this.w.onmessage = (e: MessageEvent) => {
      const { rid, ok, data, error } = e.data || {}
      const cb = this.wait.get(rid)
      if (cb) { this.wait.delete(rid); cb(ok ? data : error, !!ok) }
    }
  }
  call(cmd: string, payload?: any): Promise<any> {
    const id = this.rid++
    return new Promise((resolve, reject) => {
      this.wait.set(id, (d, ok) => ok ? resolve(d) : reject(d))
      this.w.postMessage({ cmd, payload, rid: id })
    })
  }
}

class Vertex {
  private idx: number
  private pos: Pt
  private colorIndex = 1
  private visible = true
  private diameter = 30
  private importance = 1.0
  private localDensity = 1.0
  
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
  
  // New advanced properties
  getImportance() { return this.importance }
  setImportance(imp: number) { this.importance = Math.max(0.1, Math.min(2.0, imp)) }
  getLocalDensity() { return this.localDensity }
  setLocalDensity(density: number) { this.localDensity = Math.max(0.1, density) }
  
  // Adaptive sizing based on local density
  getAdaptiveDiameter(): number {
    const baseSize = this.diameter
    const densityFactor = Math.max(0.5, Math.min(1.5, 1.0 / this.localDensity))
    const importanceFactor = this.importance
    return baseSize * densityFactor * importanceFactor
  }
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

  // New advanced features
  private spatialIndex: QuadTree | null = null
  private layoutConfig: AdvancedLayoutConfig = {
    springK: 0.18,
    repulsionStrength: 0.25,
    repulsionCutoff: 2.8,
    congestionThreshold: 10,
    congestionBeta: 0.30,
    useSpatialIndexing: true,
    maxIterations: 100,
    convergenceThreshold: 1e-6,
    magneticForces: true,
    adaptiveForces: true,
    edgeBundling: false
  }
  private lastLayoutQuality: number = 0

  // --- Enhanced public API ---
  get_label_mode() { return this._labelMode }
  setLabelMode(mode: number) {
    this._labelMode = Math.max(0, Math.min(2, Math.floor(mode)))
    this.rpc.call('set_label_mode', { mode: this._labelMode }).catch(()=>{})
  }

  // Advanced layout configuration
  setLayoutConfig(config: Partial<AdvancedLayoutConfig>) {
    this.layoutConfig = { ...this.layoutConfig, ...config }
    this._rebuildSpatialIndex()
  }

  getLayoutConfig(): AdvancedLayoutConfig {
    return { ...this.layoutConfig }
  }

  // Enhanced force calculation with spatial indexing
  calculateForces(): Pt[] {
    if (!this.layoutConfig.useSpatialIndexing || this.vertices.length < 100) {
      return this._calculateForcesBruteForce()
    }
    return this._calculateForcesWithSpatialIndex()
  }

  private _calculateForcesBruteForce(): Pt[] {
    const forces: Pt[] = new Array(this.vertices.length).fill(null).map(() => ({ x: 0, y: 0 }))
    
    for (let i = 0; i < this.vertices.length; i++) {
      if (!this.vertices[i]?.isVisible()) continue
      
      for (let j = i + 1; j < this.vertices.length; j++) {
        if (!this.vertices[j]?.isVisible()) continue
        
        const force = this._calculateRepulsionForce(i, j)
        forces[i].x += force.x; forces[i].y += force.y
        forces[j].x -= force.x; forces[j].y -= force.y
      }
    }
    
    return forces
  }

  private _calculateForcesWithSpatialIndex(): Pt[] {
    const forces: Pt[] = new Array(this.vertices.length).fill(null).map(() => ({ x: 0, y: 0 }))
    
    for (let i = 0; i < this.vertices.length; i++) {
      if (!this.vertices[i]?.isVisible()) continue
      
      const pos = this.vertices[i].getPosition()
      const neighbors = this.spatialIndex?.retrieve(pos, this.layoutConfig.repulsionCutoff * 50) || []
      
      for (const neighbor of neighbors) {
        const j = this._findVertexIndex(neighbor)
        if (j !== -1 && i !== j) {
          const force = this._calculateRepulsionForce(i, j)
          forces[i].x += force.x; forces[i].y += force.y
        }
      }
    }
    
    return forces
  }

  private _calculateRepulsionForce(i: number, j: number): Pt {
    const v1 = this.vertices[i]
    const v2 = this.vertices[j]
    if (!v1 || !v2) return { x: 0, y: 0 }
    
    const p1 = v1.getPosition()
    const p2 = v2.getPosition()
    const dx = p2.x - p1.x
    const dy = p2.y - p1.y
    const distance = Math.sqrt(dx * dx + dy * dy)
    
    if (distance < 1e-6) return { x: 0, y: 0 }
    
    const repulsionRadius = this.layoutConfig.repulsionCutoff * 50
    if (distance > repulsionRadius) return { x: 0, y: 0 }
    
    const force = this.layoutConfig.repulsionStrength * (1 - distance / repulsionRadius) / distance
    return { x: dx * force, y: dy * force }
  }

  private _findVertexIndex(pos: Point): number {
    for (let i = 0; i < this.vertices.length; i++) {
      const v = this.vertices[i]
      if (v && Math.abs(v.getPosition().x - pos.x) < 1e-6 && Math.abs(v.getPosition().y - pos.y) < 1e-6) {
        return i
      }
    }
    return -1
  }

  // Spatial indexing management
  private _rebuildSpatialIndex() {
    if (!this.layoutConfig.useSpatialIndexing) {
      this.spatialIndex = null
      return
    }
    
    const bounds = this.get_bounding_box()
    const width = bounds[2] - bounds[0]
    const height = bounds[3] - bounds[1]
    
    this.spatialIndex = new QuadTree({
      x: bounds[0],
      y: bounds[1],
      width: Math.max(width, 100),
      height: Math.max(height, 100)
    })
    
    for (const v of this.vertices) {
      if (v?.isVisible()) {
        this.spatialIndex!.insert(v.getPosition())
      }
    }
  }

  // Layout quality metrics
  calculateLayoutQuality(): number {
    let totalQuality = 0
    let count = 0
    
    // Edge length uniformity
    for (const edge of this.edges) {
      if (!edge.visible) continue
      const v1 = this.vertices[edge.u]
      const v2 = this.vertices[edge.v]
      if (v1 && v2) {
        const p1 = v1.getPosition()
        const p2 = v2.getPosition()
        const length = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        totalQuality += length
        count++
      }
    }
    
    const avgLength = count > 0 ? totalQuality / count : 0
    const variance = this._calculateEdgeLengthVariance(avgLength)
    
    // Quality score: lower variance = higher quality
    this.lastLayoutQuality = Math.max(0, 100 - variance / 10)
    return this.lastLayoutQuality
  }

  private _calculateEdgeLengthVariance(avgLength: number): number {
    let variance = 0
    let count = 0
    
    for (const edge of this.edges) {
      if (!edge.visible) continue
      const v1 = this.vertices[edge.u]
      const v2 = this.vertices[edge.v]
      if (v1 && v2) {
        const p1 = v1.getPosition()
        const p2 = v2.getPosition()
        const length = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        variance += (length - avgLength) ** 2
        count++
      }
    }
    
    return count > 0 ? variance / count : 0
  }

  getLastLayoutQuality(): number {
    return this.lastLayoutQuality
  }

  // Enhanced decluttering with quality feedback
  async declutterView(intensity = 1.0): Promise<{ changed: boolean; quality: number }> {
    const beforeQuality = this.calculateLayoutQuality()
    const raw = await this.rpc.call('declutter', { intensity })
    const res = JSON.parse(raw)
    this._applyState(res.state as State)
    
    const afterQuality = this.calculateLayoutQuality()
    const changed = beforeQuality !== afterQuality
    
    return { changed, quality: afterQuality }
  }

  // --- Original API methods (enhanced) ---
  async startBasicGraph(n: number) {
    const raw = await this.rpc.call('start', { n: Math.max(3, Math.min(10, Math.floor(n))) })
    this._applyState(JSON.parse(raw) as State)
    this._rebuildSpatialIndex()
  }

  async addRandomVertex(): Promise<[boolean, number, string?]> {
    const raw = await this.rpc.call('add_random')
    const res = JSON.parse(raw)
    this.lastAddInfo = res.info || null
    this._applyState(res.state as State)
    this._rebuildSpatialIndex()
    return [!!res.ok, Number(res.index), res.error_reason]
  }

  async addVertexBySelection(a: number, b: number): Promise<[boolean, number, string?]> {
    const raw = await this.rpc.call('add_by_selection', { a, b })
    const res = JSON.parse(raw)
    this.lastAddInfo = res.info || null
    this._applyState(res.state as State)
    this._rebuildSpatialIndex()
    return [!!res.ok, Number(res.index), res.error_reason]
  }

  async reembedTutte(_planar: boolean = true) {
    const raw = await this.rpc.call('redraw')
    this._applyState(JSON.parse(raw) as State)
    this._rebuildSpatialIndex()
  }

  async setAutoTutte(on: boolean) { await this.rpc.call('set_auto_tutte', { on: !!on }) }
  async setFinalizeMode(on: boolean) { await this.rpc.call('set_finalize', { on: !!on }) }

  async goToVertex(m: number) {
    const raw = await this.rpc.call('go_to', { m: Math.max(1, Math.floor(m)) })
    this._applyState(raw as State)
    this._rebuildSpatialIndex()
  }

  async saveToJsonObject(): Promise<any> {
    const raw = await this.rpc.call('save')
    return JSON.parse(raw)
  }

  async loadFromJsonObject(obj: any) {
    const raw = await this.rpc.call('load', { json: JSON.stringify(obj) })
    this._applyState(JSON.parse(raw) as State)
    this._rebuildSpatialIndex()
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
    
    // Rebuild spatial index after state change
    this._rebuildSpatialIndex()
    
    // notify
    if (this.onLayoutChanged) this.onLayoutChanged()
  }
}