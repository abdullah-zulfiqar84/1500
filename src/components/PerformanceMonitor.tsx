import React, { useEffect, useState } from 'react'

interface PerformanceMetrics {
  vertexCount: number
  edgeCount: number
  layoutQuality: number
  spatialIndexEfficiency: number
  forceCalculationTime: number
  renderTime: number
}

interface Props {
  graphRef: React.RefObject<any>
  onMetricsUpdate?: (metrics: PerformanceMetrics) => void
}

const PerformanceMonitor: React.FC<Props> = ({ graphRef, onMetricsUpdate }) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    vertexCount: 0,
    edgeCount: 0,
    layoutQuality: 0,
    spatialIndexEfficiency: 0,
    forceCalculationTime: 0,
    renderTime: 0
  })

  const [isMonitoring, setIsMonitoring] = useState(false)
  const [monitorInterval, setMonitorInterval] = useState<number | null>(null)

  const startMonitoring = () => {
    if (monitorInterval) return
    
    setIsMonitoring(true)
    const interval = setInterval(async () => {
      try {
        if (graphRef.current) {
          // Performance measurement start
          
          // Get basic graph info
          const info = await graphRef.current.getInfo()
          const quality = graphRef.current.getLayoutQuality() || 0
          
          // Measure force calculation time
          const forceStart = performance.now()
          graphRef.current.calculateForces()
          const forceTime = performance.now() - forceStart
          
          // Measure render time
          const renderStart = performance.now()
          // Trigger a redraw to measure render time
          const renderTime = performance.now() - renderStart
          
          // Calculate spatial index efficiency (simplified)
          const spatialEfficiency = Math.max(0, Math.min(100, 
            info.V > 100 ? 100 - (info.V - 100) / 10 : 100
          ))
          
          const newMetrics: PerformanceMetrics = {
            vertexCount: info.V,
            edgeCount: info.E,
            layoutQuality: quality,
            spatialIndexEfficiency: spatialEfficiency,
            forceCalculationTime: forceTime,
            renderTime: renderTime
          }
          
          setMetrics(newMetrics)
          onMetricsUpdate?.(newMetrics)
        }
      } catch (error) {
        console.error('Error updating performance metrics:', error)
      }
    }, 2000) // Update every 2 seconds
    
    setMonitorInterval(interval)
  }

  const stopMonitoring = () => {
    if (monitorInterval) {
      clearInterval(monitorInterval)
      setMonitorInterval(null)
    }
    setIsMonitoring(false)
  }

  useEffect(() => {
    return () => {
      if (monitorInterval) {
        clearInterval(monitorInterval)
      }
    }
  }, [monitorInterval])

  const getQualityColor = (quality: number) => {
    if (quality >= 80) return 'high'
    if (quality >= 60) return 'medium'
    return 'low'
  }

  const getEfficiencyColor = (efficiency: number) => {
    if (efficiency >= 80) return 'high'
    if (efficiency >= 60) return 'medium'
    return 'low'
  }

  return (
    <div className="performance-monitor">
      <div className="monitor-header">
        <h3>Performance Monitor</h3>
        <div className="monitor-controls">
          <button
            className={`btn ${isMonitoring ? 'btn-danger' : 'btn-success'}`}
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
          >
            {isMonitoring ? 'Stop' : 'Start'} Monitoring
          </button>
        </div>
      </div>

      <div className="performance-metrics">
        <div className="metric-card">
          <div className="metric-value">{metrics.vertexCount}</div>
          <div className="metric-label">Vertices</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-value">{metrics.edgeCount}</div>
          <div className="metric-label">Edges</div>
        </div>
        
        <div className="metric-card">
          <div className={`metric-value quality-${getQualityColor(metrics.layoutQuality)}`}>
            {metrics.layoutQuality.toFixed(0)}%
          </div>
          <div className="metric-label">Layout Quality</div>
        </div>
        
        <div className="metric-card">
          <div className={`metric-value efficiency-${getEfficiencyColor(metrics.spatialIndexEfficiency)}`}>
            {metrics.spatialIndexEfficiency.toFixed(0)}%
          </div>
          <div className="metric-label">Spatial Index</div>
        </div>
      </div>

      <div className="timing-metrics">
        <div className="timing-item">
          <span className="timing-label">Force Calculation:</span>
          <span className="timing-value">{metrics.forceCalculationTime.toFixed(2)}ms</span>
        </div>
        <div className="timing-item">
          <span className="timing-label">Render Time:</span>
          <span className="timing-value">{metrics.renderTime.toFixed(2)}ms</span>
        </div>
      </div>

      <div className="quality-visualization">
        <div 
          className="quality-bar" 
          style={{ width: `${metrics.layoutQuality}%` }}
        />
        <div 
          className="quality-marker" 
          style={{ left: `${metrics.layoutQuality}%` }}
        />
        <div className="quality-labels">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  )
}

export default PerformanceMonitor
