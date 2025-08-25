import { useEffect, useRef, useState, useCallback } from 'react'
import { CanvasGraph, type CanvasGraphHandle } from './components/CanvasGraph'
import './styles.css'


export default function App() {
  const graphRef = useRef<CanvasGraphHandle>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // UI state
  const [seed, setSeed] = useState(3)
  const [dark, setDark] = useState(false)
  const [finalize, setFinalize] = useState(false)
  const [autoTutte, setAutoTutte] = useState(true)
  const [curved, setCurved] = useState(true)
  const [shrinkOnZoom, setShrinkOnZoom] = useState(false)
  const [shrinkGamma, setShrinkGamma] = useState(1.2)
  const [selMode, setSelMode] = useState<'visual' | 'program'>('visual')
  const [goToM, setGoToM] = useState(1)
  const [stats, setStats] = useState<{ V: number, E: number, periphery: number }>({ V: 0, E: 0, periphery: 0 })

  // Chip animation helpers
  const [chipTick, setChipTick] = useState({ V: 0, E: 0, P: 0 })
  const [chipDelta, setChipDelta] = useState({ V: 0, E: 0, P: 0 }) // -1 down, 0 same, +1 up

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    const themeColorMeta = document.getElementById('theme-color') as HTMLMetaElement
    if (themeColorMeta) themeColorMeta.content = dark ? '#0e1117' : '#ffffff'
  }, [dark])

  const refreshStats = useCallback(async () => {
    try {
      const r = await graphRef.current?.getInfo()
      if (r) {
        // Trigger chip highlights on change
        if (r.V !== stats.V) {
          setChipTick(t => ({ ...t, V: t.V + 1 }))
          setChipDelta(d => ({ ...d, V: Math.sign(r.V - stats.V) }))
        }
        if (r.E !== stats.E) {
          setChipTick(t => ({ ...t, E: t.E + 1 }))
          setChipDelta(d => ({ ...d, E: Math.sign(r.E - stats.E) }))
        }
        if (r.periphery !== stats.periphery) {
          setChipTick(t => ({ ...t, P: t.P + 1 }))
          setChipDelta(d => ({ ...d, P: Math.sign(r.periphery - stats.periphery) }))
        }

        setStats(r)
        if (goToM > r.V) setGoToM(r.V || 1)
      }
    } catch (error) {
      console.error('Error refreshing stats:', error)
    }
  }, [stats.V, stats.E, stats.periphery, goToM])

  useEffect(() => { refreshStats() }, [refreshStats])

  const afterAction = (delay = 60) => window.setTimeout(refreshStats, delay)

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className={`shell ${mobileMenuOpen ? '' : 'mobile-menu-closed'}`}>
      {/* Canvas stage */}
      <main className="stage">
        <div className="stage-inner">
          <CanvasGraph ref={graphRef} />
          {/* Floating toolbar */}
          <div className="fab-col" role="toolbar" aria-label="Graph controls">
            <button className="fab" title="Zoom in (+)" aria-label="Zoom in" onClick={() => graphRef.current?.zoomIn()}>+</button>
            <button className="fab" title="Zoom out (-)" aria-label="Zoom out" onClick={() => graphRef.current?.zoomOut()}>−</button>
            <button className="fab" title="Center Graph (C)" aria-label="Center graph" onClick={() => graphRef.current?.center()}>⌖</button>
            <button className="fab" title="Declutter (L)" aria-label="Declutter view" onClick={() => graphRef.current?.declutterView()}>⋯</button>
            {/* Mobile menu toggle button */}
            <button
              className="fab mobile-menu-toggle"
              title="Toggle Controls"
              aria-label="Toggle controls panel"
              aria-expanded={mobileMenuOpen}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              style={{ display: 'none' }}
            >☰</button>
          </div>
          <div className="sel-pill" aria-live="polite">Selection: {selMode}</div>
        </div>
      </main>

      {/* Control Dock */}
      <aside className="dock">
        <div className="dock-head">
          <div className="dock-title">
            Controls
            <div className="dock-subtitle">Planar Triangulated Graph Editor</div>
          </div>
          <div className="right">
            <div className="chips">
              <div
                className={`chip ${chipDelta.V > 0 ? 'up' : chipDelta.V < 0 ? 'down' : ''}`}
                key={`V-${chipTick.V}`}
                title="Vertices"
              >V: {stats.V}</div>
              <div
                className={`chip ${chipDelta.E > 0 ? 'up' : chipDelta.E < 0 ? 'down' : ''}`}
                key={`E-${chipTick.E}`}
                title="Edges"
              >E: {stats.E}</div>
              <div
                className={`chip ${chipDelta.P > 0 ? 'up' : chipDelta.P < 0 ? 'down' : ''}`}
                key={`P-${chipTick.P}`}
                title="Periphery"
              >P: {stats.periphery}</div>
            </div>
            <label className="switch" role="switch" aria-checked={dark}>
              <input
                type="checkbox"
                checked={dark}
                onChange={e => setDark(e.target.checked)}
                aria-label="Toggle dark mode"
              />
              <span className="slider" />
              <span className="switch-label">Dark</span>
            </label>
          </div>
        </div>

        {/* FILE */}
        <details open className="card" aria-expanded="true">
          <summary role="button" aria-controls="file-section" tabIndex={0}>File</summary>
          <div id="file-section">
            <div className="row">
              <label className="sr-only" htmlFor="seed">Seed vertices</label>
              <input
                id="seed"
                className="input"
                type="number"
                min={3}
                max={10}
                value={seed}
                onChange={e => setSeed(Math.max(3, Math.min(10, parseInt(e.target.value || '3', 10))))}
                placeholder="3..10"
                aria-label="Number of seed vertices (3-10)"
              />
              <button
                className="btn primary"
                onClick={() => { graphRef.current?.startGraph(seed); afterAction() }}
                aria-label={`Create new graph with ${seed} vertices`}
              >
                New Graph
              </button>
            </div>

            <div className="row-1">
              <button
                className="btn"
                onClick={() => graphRef.current?.saveJsonAs('graph.json')}
                aria-label="Save graph as JSON file"
              >
                Save Graph
              </button>
            </div>

            {/* Clean Load Graph (hidden native input) */}
            <div className="row-1">
              <button
                className="btn"
                onClick={() => fileRef.current?.click()}
                aria-label="Load graph from JSON file"
                aria-controls="file-input"
              >
                Load Graph
              </button>
              <input
                ref={fileRef}
                type="file"
                accept="application/json"
                className="file-hidden"
                id="file-input"
                aria-label="Select JSON file to load graph"
                onChange={async (e) => {
                  const f = e.target.files?.[0]
                  if (!f) return
                  try {
                    const txt = await f.text()
                    await graphRef.current?.loadJsonObject(JSON.parse(txt))
                    refreshStats()
                  } catch {
                    alert('Invalid JSON')
                  } finally {
                    e.currentTarget.value = '' // select same file again if needed
                  }
                }}
              />
            </div>

            <div className="row-1">
              <button
                className="btn"
                onClick={() => graphRef.current?.exportPng('graph.png')}
                aria-label="Export graph as PNG image"
              >
                Export as Image
              </button>
            </div>
          </div>
        </details>

        {/* EDIT */}
        <details open className="card" aria-expanded="true">
          <summary role="button" aria-controls="edit-section" tabIndex={0}>Edit</summary>
          <div id="edit-section">
            <label className="switch-line">
              <input
                type="checkbox"
                checked={finalize}
                onChange={async e => {
                  const on = e.target.checked
                  setFinalize(on)
                  await graphRef.current?.setFinalizeMode(on)
                  refreshStats()
                }}
                aria-label="Toggle finalize mode"
              />
              <span>Finalize Mode</span>
            </label>

            <button
              className="btn"
              disabled={finalize}
              onClick={() => { graphRef.current?.addRandom(); afterAction() }}
              aria-label="Add random vertex to graph"
            >
              Add Random Vertex
            </button>

            <button
              className="btn"
              disabled={finalize}
              onClick={() => graphRef.current?.beginSelect()}
              aria-label="Add vertex by selection"
            >
              Add Vertex by Selection
            </button>

            <button
              className="btn"
              onClick={() => { graphRef.current?.redrawPlanar(); afterAction() }}
              aria-label="Redraw graph with planar layout"
            >
              Redraw (Planar)
            </button>

            <button
              className="btn"
              onClick={() => { graphRef.current?.tutteNow(); afterAction() }}
              aria-label="Apply Tutte embedding to graph"
            >
              Tutte Re-Embed Now
            </button>

            <label className="switch-line">
              <input
                type="checkbox"
                checked={autoTutte}
                onChange={e => {
                  const on = e.target.checked
                  setAutoTutte(on)
                  graphRef.current?.setAutoTutte(on)
                }}
                aria-label="Toggle automatic Tutte re-embedding"
              />
              <span>Auto Tutte Re-Embed (on add)</span>
            </label>
          </div>
        </details>

        {/* VIEW */}
        <details open className="card" aria-expanded="true">
          <summary role="button" aria-controls="view-section" tabIndex={0}>View</summary>
          <div id="view-section">
            <div className="row-1">
              <button
                className="btn"
                onClick={() => graphRef.current?.center()}
                aria-label="Center the graph in view"
              >
                Center Graph
              </button>
            </div>

            <div className="row">
              <button
                className="btn"
                onClick={() => graphRef.current?.zoomIn()}
                aria-label="Zoom in on graph"
              >
                Zoom in
              </button>
              <button
                className="btn"
                onClick={() => graphRef.current?.zoomOut()}
                aria-label="Zoom out of graph"
              >
                Zoom out
              </button>
            </div>

            <div className="row">
              <button
                className="btn"
                onClick={() => graphRef.current?.toggleLabels()}
                aria-label="Toggle visibility of vertex labels"
              >
                Toggle Labels
              </button>
              <button
                className="btn"
                onClick={() => {
                  const next = !curved
                  setCurved(next)
                  graphRef.current?.setCurvedPeriphery(next)
                }}
                aria-label="Toggle curved periphery edges"
              >
                Toggle Outer Curves
              </button>
            </div>

            <label className="switch-line">
              <input
                type="checkbox"
                checked={shrinkOnZoom}
                onChange={e => {
                  const on = e.target.checked
                  setShrinkOnZoom(on)
                  graphRef.current?.setShrinkOnZoom(on)
                }}
                aria-label="Toggle shrinking nodes when zooming"
              />
              <span>Shrink nodes on zoom</span>
            </label>

            <div className="label tiny" id="shrink-strength-label">Shrink strength</div>
            <input
              className="slider slider-shrink"
              type="range" min={0} max={100}
              value={Math.round((shrinkGamma - 1) * 100)}
              onChange={e => {
                const g = 1 + (parseInt(e.target.value || '0', 10) / 100)
                setShrinkGamma(g)
                graphRef.current?.setShrinkGamma(g)
              }}
              aria-labelledby="shrink-strength-label"
            />

            <div className="label" id="selection-mode-label">Selection CW mode</div>
            <div className="seg" role="radiogroup" aria-labelledby="selection-mode-label">
              <button
                className={`seg-btn ${selMode === 'visual' ? 'active' : ''}`}
                onClick={() => { setSelMode('visual'); graphRef.current?.setSelectionMode('visual') }}
                role="radio"
                aria-checked={selMode === 'visual'}
                tabIndex={selMode === 'visual' ? 0 : -1}
              >Visual</button>
              <button
                className={`seg-btn ${selMode === 'program' ? 'active' : ''}`}
                onClick={() => { setSelMode('program'); graphRef.current?.setSelectionMode('program') }}
                role="radio"
                aria-checked={selMode === 'program'}
                tabIndex={selMode === 'program' ? 0 : -1}
              >Program</button>
            </div>

            <div className="label" id="goto-vertex-label">Go to Vertex m</div>
            <div className="row">
              <input
                className="input"
                type="number"
                min={1}
                max={Math.max(1, stats.V)}
                value={goToM}
                onChange={e => setGoToM(Math.max(1, Math.min(stats.V || 1, parseInt(e.target.value || '1', 10))))}
                onBlur={() => graphRef.current?.goTo(goToM)}
                aria-labelledby="goto-vertex-label"
              />
              <button
                className="btn"
                onClick={() => graphRef.current?.goTo(goToM)}
                aria-label={`Go to vertex ${goToM}`}
              >
                Go
              </button>
            </div>
          </div>
        </details>

        <div className="hint" role="note" aria-label="Keyboard shortcuts and tips">
          Tips: Right‑click canvas to cancel selection. Hotkeys: N/S new • R random • A select •
          D redraw • E Tutte • L declutter • T labels • C center • G go • +/- zoom
        </div>
      </aside>
    </div>
  )
}