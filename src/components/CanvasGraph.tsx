import React, { useEffect, useImperativeHandle, useRef, useState, useCallback } from 'react'
import { Graph } from '../engine/Graph'
import { v_lerp } from '../engine/geom'
import TextType from '../TextType'
import type { GState } from '../types'

export type CanvasGraphHandle = {
  startGraph: (n: number) => void
  saveJsonAs: (filename?: string) => Promise<void>
  loadJsonObject: (obj: GState) => Promise<void>
  exportPng: (filename?: string) => Promise<void>
  addRandom: () => void
  beginSelect: () => void
  redrawPlanar: () => void
  tutteNow: () => void
  getInfo: () => Promise<{ V: number, E: number, periphery: number }>
  declutterView: () => void
  center: () => void
  zoomIn: () => void
  zoomOut: () => void
  toggleLabels: () => void
  setCurvedPeriphery: (on: boolean) => void
  setAutoTutte: (on: boolean) => void
  setFinalizeMode: (on: boolean) => void
  setShrinkOnZoom: (on: boolean) => void
  setShrinkGamma: (g: number) => void
  setSelectionMode: (mode: 'visual' | 'program') => void
  goTo: (m: number) => void
}

type Props = Record<string, never>
type Pt = { x: number, y: number }

const NEW_GRAPH_ANIM_MS = 600

export const CanvasGraph = React.forwardRef<CanvasGraphHandle, Props>((_props, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [graph] = useState(() => new Graph())

  // Intro overlay
  const [showTitle, setShowTitle] = useState(true)
  const [introHiding, setIntroHiding] = useState(false)
  const hideIntro = (ms = 420) => {
    if (!showTitle || introHiding) return
    setIntroHiding(true)
    window.setTimeout(() => setShowTitle(false), ms)
  }

  // Curved periphery toggle
  const [curvedPeriphery, setCurvedPeriphery] = useState(true)
  const curvedRef = useRef(true)

  // Optional grid toggle
  const gridRef = useRef(false)

  // camera
  const cam = useRef({ x: 0, y: 0, scale: 1 })
  const isDown = useRef(false)
  const dragging = useRef(false)
  const downAt = useRef({ x: 0, y: 0 })
  const last = useRef({ x: 0, y: 0 })
  const selecting = useRef<{ vp: number | null }>({ vp: null })

  // node sizing
  const shrinkOnZoom = useRef(true)
  const shrinkGamma = useRef(1.2)

  // selection mode and finalize
  const selectionMode = useRef<'visual' | 'program'>('visual')
  const finalizeRef = useRef(false)

  // animation
  const animRef = useRef<{ pre: Record<number, Pt>, post: Record<number, Pt>, t0: number, dur: number } | null>(null)

  // HUD
  const [hud, setHud] = useState<string>('')
  const hudTimer = useRef<number | null>(null)
  function showHud(text: string, ms = 2600) {
    setHud(text)
    if (hudTimer.current) window.clearTimeout(hudTimer.current)
    hudTimer.current = window.setTimeout(() => setHud(''), ms)
  }

  const draw = useCallback(() => {
    try {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d'); if (!ctx) return
      
      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth || 300, h = canvas.clientHeight || 150
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr; canvas.height = h * dpr
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      }
      ctx.clearRect(0, 0, w, h)

      drawGrid(ctx, w, h)
    } catch {
      // Silently continue if drawing setup fails
      return
    }

    let peri: number[] = []
    let nPer = 0
    let orientCCW = true

    try {
      const canvas = canvasRef.current; if (!canvas) return
      const ctx = canvas.getContext('2d'); if (!ctx) return
      
      const edgeColor = getComputedStyle(canvas).getPropertyValue('--edge-color').trim() || '#3c3c3c'
      ctx.lineCap = 'round'; ctx.lineJoin = 'round'
      const scale = cam.current?.scale || 1
      const verticesLength = Array.isArray(graph?.vertices) ? graph.vertices.length : 0
      const edgeWidth = Math.max(0.25, 2.0 / (1.0 + 0.9 * scale + 0.00012 * verticesLength))
      ctx.strokeStyle = edgeColor; ctx.lineWidth = edgeWidth

      peri = graph.periphery?.getIndices?.() ?? []
      nPer = peri.length
      try {
        let A = 0
        for (let i = 0; i < nPer; i++) {
          const aV = graph?.vertices?.[peri[i]]
          const bV = graph?.vertices?.[peri[(i + 1) % nPer]]
          if (!aV || !bV) continue
          const a = aV.getPosition(), b = bV.getPosition()
          A += a.x * b.y - b.x * a.y
        }
        orientCCW = A > 0
      } catch {
        orientCCW = true
      }
    } catch {
      // Silently continue if edge setup fails
      return
    }

    // Edges
    try {
      const canvas = canvasRef.current; if (!canvas) return
      const ctx = canvas.getContext('2d'); if (!ctx) return
      const w = canvas.clientWidth || 300, h = canvas.clientHeight || 150
      ctx.beginPath()
      for (const e of graph.edges || []) {
        try {
          if (!e || e.visible === false) continue
          if (!graph?.vertices?.[e.u] || !graph?.vertices?.[e.v]) continue
          const p1 = graph.vertices[e.u].getPosition()
          const p2 = graph.vertices[e.v].getPosition()
          if (!p1 || !p2) continue
          const iu = peri.indexOf(e.u)
          let isPeriEdge = false
          if (nPer >= 2 && iu >= 0) {
            isPeriEdge = (peri[(iu + 1) % nPer] === e.v) || (peri[(iu - 1 + nPer) % nPer] === e.v)
          }
          if (isPeriEdge && curvedRef.current) {
            const nextOfU = peri[(iu + 1) % nPer], isUvOrder = (nextOfU === e.v)
            const p1d = isUvOrder ? p1 : p2, p2d = isUvOrder ? p2 : p1
            const s1 = worldToScreen(p1d.x, p1d.y), s2 = worldToScreen(p2d.x, p2d.y)
            const dx = s2.x - s1.x, dy = s2.y - s1.y, L = Math.hypot(dx, dy) || 1
            const mx = (s1.x + s2.x) / 2, my = (s1.y + s2.y) / 2
            let nx = orientCCW ? (dy / L) : (-dy / L), ny = orientCCW ? (-dx / L) : (dx / L)
            const c = graph.get_center?.() ?? { x: 0, y: 0 }, sc = worldToScreen(c.x, c.y)
            if (nx * (mx - sc.x) + ny * (my - sc.y) < 0) { nx = -nx; ny = -ny }
            const viewMin = Math.min(w, h)
            let sagPx = L * 0.07
            if (L > 0.6 * viewMin) sagPx *= 0.75
            if (L > 0.9 * viewMin) sagPx *= 0.65
            sagPx = Math.min(sagPx, 0.12 * viewMin)
            const cx = mx + nx * (2 * sagPx), cy = my + ny * (2 * sagPx)
            ctx.moveTo(s1.x, s1.y); ctx.quadraticCurveTo(cx, cy, s2.x, s2.y)
          } else {
            const s1 = worldToScreen(p1.x, p1.y), s2 = worldToScreen(p2.x, p2.y)
            ctx.moveTo(s1.x, s1.y); ctx.lineTo(s2.x, s2.y)
          }
        } catch {
          // Silently continue if individual edge rendering fails
          continue
        }
      }
      ctx.stroke()
    } catch {
      // Silently continue if edge rendering fails
      return
    }

    // Vertices
    try {
      const canvas = canvasRef.current; if (!canvas) return
      const ctx = canvas.getContext('2d'); if (!ctx) return
      const labelMode = graph.get_label_mode() ?? 0
      const paletteOutline = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f']
      const paletteFill = ['#ff6b6b', '#48dbfb', '#1dd1a1', '#feca57']
      for (const v of graph.vertices || []) {
        try {
          if (!v || typeof v.isVisible !== 'function' || !v.isVisible()) continue
          const p = v.getPosition(); if (!p) continue
          const s = worldToScreen(p.x, p.y)
          const baseD = typeof v.getDiameter === 'function' ? v.getDiameter() : 30
          const scale = cam.current?.scale || 1
          const gamma = shrinkOnZoom.current ? Math.max(1.0, Math.min(2.0, shrinkGamma.current || 1.0)) : 1.0
          const nodePx = Math.max(6, Math.min(44, (baseD * scale) / Math.max(Math.pow(scale, gamma), 1)))
          const d = nodePx
          const colorIdx = (typeof v.getColorIndex === 'function' ? v.getColorIndex() : 1) - 1
          const colorIdxSafe = ((colorIdx % 4) + 4) % 4
          const fill = labelMode === 1 ? '#ffffff' : paletteFill[colorIdxSafe]
          const stroke = labelMode === 1 ? '#000000' : paletteOutline[colorIdxSafe]
          const ctx2 = ctx
          ctx2.fillStyle = fill; ctx2.strokeStyle = stroke; ctx2.lineWidth = 2
          ctx2.beginPath(); ctx2.arc(s.x, s.y, d / 2, 0, Math.PI * 2); ctx2.fill(); ctx2.stroke()
          if (labelMode === 1 || labelMode === 2) {
            try {
              ctx2.fillStyle = '#000000'
              ctx2.font = `${Math.max(1, Math.round(0.6 * d))}px Arial`
              ctx2.textAlign = 'center'; ctx2.textBaseline = 'middle'
              const index = typeof v.getIndex === 'function' ? v.getIndex() : 0
              ctx2.fillText(String(index + 1), s.x, s.y)
            } catch {
              // Silently continue if text rendering fails
            }
          }
        } catch {
          // Silently continue if individual vertex rendering fails
          continue
        }
      }
    } catch {
      // Silently continue if vertex rendering fails
    }
  }, [graph, cam, curvedRef, shrinkOnZoom, shrinkGamma, worldToScreen, drawGrid])

  const center = useCallback((animMs = 0) => {
    const [minx, miny, maxx, maxy] = graph.get_bounding_box()
    const canvas = canvasRef.current
    if (!canvas) return

    const w = canvas.clientWidth || canvas.width
    const h = canvas.clientHeight || canvas.height
    const pw = maxx - minx, ph = maxy - miny
    if (w < 2 || h < 2 || pw < 1e-6 || ph < 1e-6) return

    const basePadPx = Math.max(40, Math.min(80, Math.min(w, h) * 0.06))
    const s1 = Math.min((w - 2 * basePadPx) / pw, (h - 2 * basePadPx) / ph)
    const extraPadPx = curvedRef.current ? maxPeripherySagPx(s1, w, h) : 0
    const padPx = basePadPx + extraPadPx
    const targetScale = Math.min((w - 2 * padPx) / pw, (h - 2 * padPx) / ph)
    const targetX = (minx + maxx) / 2 - w / (2 * targetScale)
    const targetY = (miny + maxy) / 2 - h / (2 * targetScale)

    if (!animMs) {
      cam.current = { x: targetX, y: targetY, scale: targetScale }
      draw()
      return
    }
    const a0 = { ...cam.current }, a1 = { x: targetX, y: targetY, scale: targetScale }
    const t0 = performance.now()
    const stepCam = (now: number) => {
      const t = Math.min(1, (now - t0) / animMs)
      cam.current = {
        x: a0.x + (a1.x - a0.x) * t,
        y: a0.y + (a1.y - a0.y) * t,
        scale: a0.scale + (a1.scale - a0.scale) * t,
      }
      draw()
      if (t < 1) requestAnimationFrame(stepCam)
    }
    requestAnimationFrame(stepCam)
  }, [graph, curvedRef, draw, maxPeripherySagPx])

  const zoomBy = useCallback((f: number) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const MIN_SCALE = 1e-4, MAX_SCALE = 1e6
    const rect = canvas.getBoundingClientRect()
    const cx = rect.width / 2, cy = rect.height / 2
    const before = screenToWorld(cx, cy)
    cam.current.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, cam.current.scale * f))
    const after = screenToWorld(cx, cy)
    cam.current.x += (before.x - after.x)
    cam.current.y += (before.y - after.y)
    draw()
  }, [cam, screenToWorld, draw])

  useEffect(() => {
    curvedRef.current = curvedPeriphery
  }, [curvedPeriphery])

  useEffect(() => {
    // layout changes from engine
    graph.onLayoutChanged = () => {
      if (!animRef.current && !isDown.current && !dragging.current) center(0)
      else draw()
    }

    // initial seed with spawn animation
    void graph.startBasicGraph(3).then(() => {
      hideIntro(420) // fade out intro smoothly
      const post = snapshot()
      animateSpawnFromCenter(post, 0.08, NEW_GRAPH_ANIM_MS)
    })

    return () => {
      if (hudTimer.current) window.clearTimeout(hudTimer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [center, draw])

  useEffect(() => {
    const onResize = () => { if (!animRef.current && !isDown.current && !dragging.current) center(0) }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [center])

  // Keyboard navigation
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    canvas.setAttribute('tabindex', '0')
    canvas.setAttribute('role', 'application')
    canvas.setAttribute('aria-label', 'Graph canvas. Use arrow keys to navigate, + and - to zoom, C to center.')

    const handleKeyDown = (e: KeyboardEvent) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', '+', '-', '=', 'c', 'C', 'Escape'].includes(e.key)) {
        e.preventDefault()
      }

      const panStep = 50 / cam.current.scale

      switch (e.key) {
        case 'ArrowUp':
          cam.current.y -= panStep; draw(); break
        case 'ArrowDown':
          cam.current.y += panStep; draw(); break
        case 'ArrowLeft':
          cam.current.x -= panStep; draw(); break
        case 'ArrowRight':
          cam.current.x += panStep; draw(); break
        case '+':
        case '=':
          zoomBy(1.2); break
        case '-':
          zoomBy(1/1.2); break
        case 'c':
        case 'C':
          center(); break
        case 'Escape':
          selecting.current.vp = null
          showHud('Selection cancelled')
          break
      }
    }

    canvas.addEventListener('keydown', handleKeyDown)
    return () => canvas.removeEventListener('keydown', handleKeyDown)
  }, [draw, zoomBy, center])

  // Fallback declutter helpers
  function buildAdj(): Map<number, Set<number>> {
    const adj = new Map<number, Set<number>>()
    const edges = graph.edges || []
    for (const e of edges) {
      const u = e.u, v = e.v
      if (!adj.has(u)) adj.set(u, new Set())
      if (!adj.has(v)) adj.set(v, new Set())
      adj.get(u)!.add(v); adj.get(v)!.add(u)
    }
    return adj
  }
  function fallbackDeclutter(rounds = 2) {
    const per = graph.periphery?.getIndices?.() || []
    const perSet = new Set<number>(per)
    const adj = buildAdj()
    for (let r = 0; r < rounds; r++) {
      for (let i = 0; i < graph.vertices.length; i++) {
        if (perSet.has(i)) continue
        const v = graph.vertices[i]; if (!v.isVisible()) continue
        const neigh = Array.from(adj.get(i) || [])
        if (!neigh.length) continue
        let sx = 0, sy = 0, c = 0
        for (const j of neigh) {
          const p = graph.vertices[j].getPosition()
          sx += p.x; sy += p.y; c++
        }
        const old = v.getPosition()
        const nx = c > 0 ? sx / c : old.x
        const ny = c > 0 ? sy / c : old.y
        v.setPosition({ x: 0.68 * nx + 0.32 * old.x, y: 0.68 * ny + 0.32 * old.y })
      }
    }
  }

  // Wait until any in-progress animation finishes
  function waitForIdle(): Promise<void> {
    return new Promise(resolve => {
      const tick = () => (animRef.current ? requestAnimationFrame(tick) : resolve())
      tick()
    })
  }

  useImperativeHandle(ref, () => ({
    startGraph(n: number) {
      void graph.startBasicGraph(n).then(() => {
        hideIntro(420)
        const post = snapshot()
        animateSpawnFromCenter(post, 0.08, NEW_GRAPH_ANIM_MS)
        showHud('New graph started')
      })
    },

    async saveJsonAs(filename = 'graph.json') {
      const data = await graph.saveToJsonObject()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a'); a.href = url; a.download = filename; a.click()
      URL.revokeObjectURL(url)
      showHud('Saved graph.json')
    },

    async loadJsonObject(obj: GState) {
      hideIntro(300)
      await graph.loadFromJsonObject(obj)
      center(0); draw(); showHud('Graph loaded')
    },

    async exportPng(filename = 'graph.png') {
      const canvas = canvasRef.current!
      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth, h = canvas.clientHeight
      const tmp = document.createElement('canvas')
      tmp.width = Math.max(1, Math.round(w * dpr));
      tmp.height = Math.max(1, Math.round(h * dpr))
      const ctx = tmp.getContext('2d')!
      ctx.drawImage(canvas, 0, 0, tmp.width, tmp.height)
      const url = tmp.toDataURL('image/png')
      const a = document.createElement('a'); a.href = url; a.download = filename; a.click()
      showHud('Exported image')
    },

    addRandom() {
      void (async () => {
        if (finalizeRef.current) return
        await waitForIdle() // don’t ignore clicks during animation
        const pre = snapshot()
        const [ok] = await graph.addRandomVertex()
        if (!ok) { showHud('No valid periphery arc'); return }
        const post = snapshot()
        const info = graph.lastAddInfo
        if (info && info.spawn_pos && pre[info.index] === undefined) pre[info.index] = info.spawn_pos
        animate(pre, post, 420)
      })()
    },

    beginSelect() {
      if (finalizeRef.current) return
      selecting.current.vp = null
      showHud('Select Vp (periphery), then Vq')
    },

    redrawPlanar() {
      const pre = snapshot()
      void graph.reembedTutte(true).then(() => {
        const post = snapshot()
        animate(pre, post, 420)
        showHud('Redraw complete')
      })
    },

    tutteNow() {
      const pre = snapshot()
      void graph.reembedTutte(true).then(() => {
        const post = snapshot()
        animate(pre, post, 420)
        showHud('Tutte re-embed done')
      })
    },

    async getInfo() {
      try {
        const s = graph.getStats?.() || {
          total_vertices: graph.vertices?.length || 0,
          edges: graph.edges?.length ?? 0,
          periphery_size: graph.periphery?.getIndices?.().length ?? 0
        }
        const V = s.total_vertices || 0
        const E = s.edges || 0
        const periphery = s.periphery_size || 0
        const text = `V=${V}, E=${E}, periphery=${periphery}`
        showHud(text, 3200)
        return { V, E, periphery }
      } catch (error) {
        console.error('Error getting graph info:', error)
        return { V: 0, E: 0, periphery: 0 }
      }
    },

    declutterView() {
      if (animRef.current) return
      const pre = snapshot()
      void (async () => {
        let changed = false
        if (typeof graph.declutterView === 'function') {
          changed = await graph.declutterView()
        } else if (typeof graph.reembedTutte === 'function') {
          await graph.reembedTutte(true)
          changed = true
        } else {
          fallbackDeclutter(2)
          changed = true
        }
        const post = snapshot()
        if (changed) animate(pre, post, 360)
        showHud('Declutter applied')
      })()
    },

    center: () => { center(0) },
    zoomIn:  () => zoomBy(1.15),
    zoomOut: () => zoomBy(1 / 1.15),

    toggleLabels() {
      const next = (graph.get_label_mode() + 1) % 3
      graph.setLabelMode(next as 0 | 1 | 2)
      draw()
      showHud('Toggled labels')
    },

    setCurvedPeriphery(on: boolean) {
      curvedRef.current = on
      setCurvedPeriphery(on)
      draw()
      showHud(`Outer curves: ${on ? 'ON' : 'OFF'}`)
    },

    setAutoTutte(on: boolean) {
      void graph.setAutoTutte(on)
      showHud(`Auto Tutte: ${on ? 'ON' : 'OFF'}`)
    },

    setFinalizeMode(on: boolean) {
      finalizeRef.current = !!on
      void (async () => {
        await graph.setFinalizeMode(!!on)
        if (on) {
          await graph.goToVertex(4)
          showHud('Finalize: ON (G4)')
        } else {
          const total = graph.getStats?.().total_vertices ?? graph.vertices.length
          await graph.goToVertex(total)
          showHud('Finalize: OFF')
        }
        center(0); draw()
      })()
    },

    setShrinkOnZoom(on: boolean) { shrinkOnZoom.current = on; draw() },
    setShrinkGamma(g: number) { shrinkGamma.current = Math.max(1.0, Math.min(2.0, g)); draw() },
    setSelectionMode(mode: 'visual' | 'program') { selectionMode.current = mode; showHud(`Selection: ${mode}`) },

    goTo(m: number) {
      void graph.goToVertex(Math.max(1, Math.floor(m))).then(() => {
        draw(); center(0)
        showHud(`Showing 1..${Math.max(1, Math.floor(m))}`)
      })
    },
  }))

  useEffect(() => { center(0) }, [curvedPeriphery, center])

  function snapshot() {
    const snap: Record<number, Pt> = {}
    for (const v of graph.vertices) if (v) snap[v.getIndex()] = { ...v.getPosition() }
    return snap
  }

  function animateSpawnFromCenter(post: Record<number, Pt>, spawnRatio = 0.08, duration = NEW_GRAPH_ANIM_MS) {
    const c = graph.get_center?.() ?? { x: 0, y: 0 }
    const pre: Record<number, Pt> = {}
    for (const kStr of Object.keys(post)) {
      const k = Number(kStr)
      const p = post[k]
      pre[k] = { x: c.x + (p.x - c.x) * spawnRatio, y: c.y + (p.y - c.y) * spawnRatio }
    }
    animate(pre, post, duration)
  }

  function animate(pre: Record<number, Pt>, post: Record<number, Pt>, dur = 420) {
    for (const kStr of Object.keys(pre)) {
      const k = Number(kStr)
      graph.vertices[k]?.setPosition(pre[k])
    }
    draw()
    animRef.current = { pre, post, t0: performance.now(), dur }
    requestAnimationFrame(step)
  }

  function step(now: number) {
    if (!animRef.current) return
    const { pre, post, t0, dur } = animRef.current
    const t = Math.min(1, (now - t0) / dur)
    const keys = new Set([...Object.keys(pre), ...Object.keys(post)].map(Number))
    for (const k of keys) {
      const a = pre[k] ?? post[k], b = post[k] ?? pre[k]
      const p = v_lerp(a, b, t)
      graph.vertices[k]?.setPosition(p)
    }
    draw()
    if (t < 1) requestAnimationFrame(step)
    else { animRef.current = null; center(600) }
  }

  // Fit + rendering
  const maxPeripherySagPx = useCallback((scaleEstimate: number, w: number, h: number): number => {
    const peri = graph.periphery.getIndices()
    if (peri.length < 2) return 0
    const viewMin = Math.min(w, h)
    let maxSag = 0
    for (let i = 0; i < peri.length; i++) {
      const a = graph.vertices[peri[i]].getPosition()
      const b = graph.vertices[peri[(i + 1) % peri.length]].getPosition()
      const Lworld = Math.hypot(b.x - a.x, b.y - a.y)
      const Lpx = Lworld * scaleEstimate
      let sagPx = 0.07 * Lpx
      if (Lpx > 0.6 * viewMin) sagPx *= 0.75
      if (Lpx > 0.9 * viewMin) sagPx *= 0.65
      sagPx = Math.min(sagPx, 0.12 * viewMin)
      if (sagPx > maxSag) maxSag = sagPx
    }
    return Math.max(12, maxSag * 1.15)
  }, [graph])

  const worldToScreen = useCallback((x: number, y: number) => {
    try {
      const { scale, x: ox, y: oy } = cam.current || { scale: 1, x: 0, y: 0 }
      const validScale = Math.max(1e-9, scale)
      return { x: (x - ox) * validScale, y: (y - oy) * validScale }
    } catch (error) {
      console.error('Error in worldToScreen:', error)
      return { x: 0, y: 0 }
    }
  }, [cam])
  
  const screenToWorld = useCallback((x: number, y: number) => {
    try {
      const { scale, x: ox, y: oy } = cam.current || { scale: 1, x: 0, y: 0 }
      const validScale = Math.max(1e-9, scale)
      return { x: x / validScale + ox, y: y / validScale + oy }
    } catch (error) {
      console.error('Error in screenToWorld:', error)
      return { x: 0, y: 0 }
    }
  }, [cam])

  function angleUp(p: Pt, c: Pt) { return Math.atan2(-(p.y - c.y), (p.x - c.x)) }
  function dtheta(a: number, b: number) {
    let d = a - b; while (d <= -Math.PI) d += 2 * Math.PI; while (d > Math.PI) d -= 2 * Math.PI; return d
  }
  function visualCwIsStoredNext(vp: number): boolean {
    const [prev, next] = graph.periphery?.neighborsOnPeriphery?.(vp) ?? [null, null]
    if (prev == null || next == null) return true
    const c = graph.get_center?.() ?? { x: 0, y: 0 }
    const pv = graph.vertices[vp].getPosition()
    const pn = graph.vertices[next].getPosition()
    const tv = angleUp(pv, c), tn = angleUp(pn, c)
    return dtheta(tn, tv) < 0.0
  }
  function mapSelectionToProgramCw(a: number, b: number): [number, number] {
    return (selectionMode.current === 'program') ? [a, b] : (visualCwIsStoredNext(a) ? [a, b] : [b, a])
  }

  // Grid rendering
  const drawGrid = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number) => {
    if (!gridRef.current) return
    const canvas = canvasRef.current
    if (!canvas) return

    const cs = getComputedStyle(canvas)
    const minor = cs.getPropertyValue('--grid-minor').trim() || 'rgba(0,0,0,0.06)'
    const major = cs.getPropertyValue('--grid-major').trim() || 'rgba(0,0,0,0.10)'

    const { scale, x: ox, y: oy } = cam.current
    const targetPx = 64
    let stepW = targetPx / Math.max(scale, 1e-9)
    const pow10 = Math.pow(10, Math.floor(Math.log10(stepW)))
    const k = stepW / pow10
    const mul = k < 2 ? 1 : (k < 5 ? 2 : 5)
    stepW = mul * pow10

    const minX = ox
    const maxX = ox + w / Math.max(scale, 1e-9)
    const minY = oy
    const maxY = oy + h / Math.max(scale, 1e-9)

    const startX = Math.floor(minX / stepW) * stepW
    const startY = Math.floor(minY / stepW) * stepW

    const maxLines = 1200
    ctx.lineWidth = 1

    let count = 0
    for (let x = startX; x <= maxX && count < maxLines; x += stepW, count++) {
      const sx = Math.round((x - ox) * scale) + 0.5
      ctx.beginPath()
      ctx.moveTo(sx, 0)
      ctx.lineTo(sx, h)
      ctx.strokeStyle = (count % 5 === 0) ? major : minor
      ctx.stroke()
    }

    count = 0
    for (let y = startY; y <= maxY && count < maxLines; y += stepW, count++) {
      const sy = Math.round((y - oy) * scale) + 0.5
      ctx.beginPath()
      ctx.moveTo(0, sy)
      ctx.lineTo(w, sy)
      ctx.strokeStyle = (count % 5 === 0) ? major : minor
      ctx.stroke()
    }
  }, [gridRef, cam])

  // Events
  useEffect(() => {
    const canvas = canvasRef.current!
    const onContextMenu = (e: MouseEvent) => { e.preventDefault(); selecting.current.vp = null }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const { offsetX, offsetY, deltaY } = e
      const s = Math.exp(-deltaY / 300)
      const { x: wx, y: wy } = screenToWorld(offsetX, offsetY)
      cam.current.scale *= s
      const { x: wx2, y: wy2 } = screenToWorld(offsetX, offsetY)
      cam.current.x += wx - wx2; cam.current.y += wy - wy2
      draw()
    }
    
    let lastPinchDistance = 0
    const getFingerDistance = (e: TouchEvent): number => {
      if (e.touches.length < 2) return 0
      const dx = e.touches[0].clientX - e.touches[1].clientX
      const dy = e.touches[0].clientY - e.touches[1].clientY
      return Math.sqrt(dx * dx + dy * dy)
    }
    const onDown = (e: MouseEvent) => { isDown.current = true; dragging.current = false; downAt.current = { x: e.clientX, y: e.clientY }; last.current = { x: e.clientX, y: e.clientY } }
    const onMove = (e: MouseEvent) => {
      if (!isDown.current) return
      const moved = Math.hypot(e.clientX - downAt.current.x, e.clientY - downAt.current.y)
      if (!dragging.current && moved > 3) dragging.current = true
      if (dragging.current) {
        const dx = (e.clientX - last.current.x) / cam.current.scale, dy = (e.clientY - last.current.y) / cam.current.scale
        cam.current.x -= dx; cam.current.y -= dy; last.current = { x: e.clientX, y: e.clientY }; draw()
      }
    }
    const onUp = (e: MouseEvent) => {
      const wasDragging = dragging.current
      isDown.current = false
      dragging.current = false

      const canvasEl = canvasRef.current
      if (!canvasEl) return
      const rect = canvasEl.getBoundingClientRect()
      const offsetX = e.clientX - rect.left
      const offsetY = e.clientY - rect.top

      if (!wasDragging) {
        if (selecting.current.vp !== null) {
          const pre = snapshot()
          const vq = findVertexAt(offsetX, offsetY, true)
          if (vq !== null) {
            const [a, b] = mapSelectionToProgramCw(selecting.current.vp, vq)
            void (async () => {
              const [ok] = await graph.addVertexBySelection(a, b)
              if (ok) {
                const post = snapshot()
                const info = graph.lastAddInfo
                if (info && info.spawn_pos && pre[info.index] === undefined) pre[info.index] = info.spawn_pos
                animate(pre, post, 420)
              }
            })()
          }
          selecting.current.vp = null
        } else {
          const vp = findVertexAt(offsetX, offsetY, true)
          if (vp !== null) selecting.current.vp = vp
        }
      }
    }
    
    // Touch events
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        lastPinchDistance = getFingerDistance(e)
        e.preventDefault()
        return
      }
      if (e.touches.length !== 1) return
      e.preventDefault()
      isDown.current = true
      dragging.current = false
      downAt.current = { x: e.touches[0].clientX, y: e.touches[0].clientY }
      last.current = { x: e.touches[0].clientX, y: e.touches[0].clientY }
    }
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        const currentDistance = getFingerDistance(e)
        if (lastPinchDistance > 0) {
          const pinchRatio = currentDistance / lastPinchDistance
          if (pinchRatio !== 1) {
            const rect = canvas.getBoundingClientRect()
            const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left
            const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top
            const before = screenToWorld(midX, midY)
            cam.current.scale = Math.min(1e6, Math.max(1e-4, cam.current.scale * pinchRatio))
            const after = screenToWorld(midX, midY)
            cam.current.x += (before.x - after.x)
            cam.current.y += (before.y - after.y)
            draw()
          }
        }
        lastPinchDistance = currentDistance
        e.preventDefault()
        return
      }
      if (!isDown.current || e.touches.length !== 1) return
      e.preventDefault()
      const moved = Math.hypot(e.touches[0].clientX - downAt.current.x, e.touches[0].clientY - downAt.current.y)
      if (!dragging.current && moved > 3) dragging.current = true
      if (dragging.current) {
        const dx = (e.touches[0].clientX - last.current.x) / cam.current.scale
        const dy = (e.touches[0].clientY - last.current.y) / cam.current.scale
        cam.current.x -= dx; cam.current.y -= dy
        last.current = { x: e.touches[0].clientX, y: e.touches[0].clientY }
        draw()
      }
    }
    const onTouchEnd = (e: TouchEvent) => {
      const wasDragging = dragging.current
      isDown.current = false
      dragging.current = false
      if (!wasDragging && e.changedTouches.length === 1) {
        const touch = e.changedTouches[0]
        const rect = canvas.getBoundingClientRect()
        const offsetX = touch.clientX - rect.left
        const offsetY = touch.clientY - rect.top
        if (selecting.current.vp !== null) {
          const pre = snapshot()
          const vq = findVertexAt(offsetX, offsetY, true)
          if (vq !== null) {
            const [a, b] = mapSelectionToProgramCw(selecting.current.vp, vq)
            void (async () => {
              const [ok] = await graph.addVertexBySelection(a, b)
              if (ok) {
                const post = snapshot()
                const info = graph.lastAddInfo
                if (info && info.spawn_pos && pre[info.index] === undefined) pre[info.index] = info.spawn_pos
                animate(pre, post, 420)
              }
            })()
          }
          selecting.current.vp = null
        } else {
          const vp = findVertexAt(offsetX, offsetY, true)
          if (vp !== null) selecting.current.vp = vp
        }
      }
    }
    
    canvas.addEventListener('contextmenu', onContextMenu)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('mousedown', onDown)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    canvas.addEventListener('touchstart', onTouchStart, { passive: false })
    canvas.addEventListener('touchmove', onTouchMove, { passive: false })
    canvas.addEventListener('touchend', onTouchEnd)
    
    return () => {
      canvas.removeEventListener('contextmenu', onContextMenu)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('mousedown', onDown)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      canvas.removeEventListener('touchstart', onTouchStart)
      canvas.removeEventListener('touchmove', onTouchMove)
      canvas.removeEventListener('touchend', onTouchEnd)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function findVertexAt(offsetX: number, offsetY: number, peripheryOnly = false): number | null {
    try {
      if (!graph?.vertices?.length) return null
      const p = screenToWorld(offsetX, offsetY)
      const perSet = new Set(graph.periphery?.getIndices?.() ?? [])
      const scale = Math.max(1e-9, cam.current?.scale || 1)
      const gamma = shrinkOnZoom.current ? Math.max(1.0, Math.min(2.0, shrinkGamma.current || 1.0)) : 1.0

      for (let i = graph.vertices.length - 1; i >= 0; i--) {
        const gv = graph.vertices[i]
        if (!gv) continue
        try {
          const pos = gv.getPosition()
          if (!pos || !gv.isVisible()) continue
          if (peripheryOnly && !perSet.has(i)) continue
          const baseD = gv.getDiameter?.() || 30
          const rWorld = (baseD / 2) / Math.pow(scale, gamma) * 1.2 // easier clicking
          const dx = pos.x - p.x, dy = pos.y - p.y
          if (Math.hypot(dx, dy) <= rWorld) return i
        } catch {
          // Silently continue if individual vertex processing fails
          continue
        }
      }
    } catch {
      // Silently continue if vertex finding fails
    }
    return null
  }

  const getAccessibleDescription = () => {
    const vertexCount = graph.vertices.length;
    const edgeCount = graph.edges?.length || 0;
    const peripheryCount = graph.periphery?.getIndices?.().length || 0;
    return `Graph with ${vertexCount} vertices and ${edgeCount} edges. ${peripheryCount} vertices on periphery. ${selecting.current.vp !== null ? 'Currently in selection mode.' : ''}`;
  };

  return (
    <div className="canvas-wrap">
      <canvas 
        ref={canvasRef} 
        className="graph-canvas"
        aria-label="Graph visualization"
        role="application"
        tabIndex={0}
      />
      {/* Centered title overlay — fades out on first graph */}
      {showTitle && (
        <div
          className="canvas-center-title"
          aria-hidden="true"
          data-hide={introHiding ? '1' : '0'}
        >
          <TextType
            text="Planar Triangulated Graph Editor"
            typingSpeed={32}
            pauseDuration={3500}
            showCursor={false}
          />
        </div>
      )}
      {/* HUD appears only when showHud() is called */}
      <div className="hud" data-show={hud ? '1' : '0'}>{hud}</div>
      <div className="sr-only" aria-live="polite">
        {getAccessibleDescription()}
      </div>
    </div>
  )
})

CanvasGraph.displayName = 'CanvasGraph'