# public/py/bridge.py
import json
from graph import Graph
from qtcore_shim import QPointF  # noqa

_g = Graph()
_g.set_auto_tutte_reembed(True)  # use pure-Python Tutte (GS) in browser

# Advanced layout configuration
_layout_config = {
    "spring_k": 0.18,
    "repulsion_strength": 0.25,
    "repulsion_cutoff": 2.8,
    "congestion_threshold": 10,
    "congestion_beta": 0.30,
    "use_spatial_indexing": True,
    "max_iterations": 100,
    "convergence_threshold": 1e-6,
    "magnetic_forces": True,
    "adaptive_forces": True,
    "edge_bundling": False
}

def _pt(p):
    return {"x": float(p.x()), "y": float(p.y())}

def _state_dict():
    verts = []
    for v in _g.getVertices():
        if v is None: 
            continue
        p = v.getPosition()
        verts.append({
            "index": v.getIndex(),
            "x": float(p.x()),
            "y": float(p.y()),
            "color": int(v.getColorIndex()),
            "visible": bool(v.isVisible()),
        })
    edges = [{"u": e.getStartVertex().getIndex(),
              "v": e.getEndVertex().getIndex(),
              "visible": bool(e.isVisible())} for e in _g.getEdges()]
    return {
        "vertices": verts,
        "edges": edges,
        "periphery": list(_g.getPeriphery()),
        "meta": _g.get_stats(),
        "labelMode": int(_g.get_label_mode()),
    }

def _post_layout_clean(intensity: float = 1.15):
    """
    Gentle, planarity-safe spacing:
    - positive-weight Tutte relax (shrinks long, grows short)
    - vertex-vertex min-distance separation
    - vertex-edge clearance
    """
    try:
        _g.declutter_for_view(float(intensity))
    except Exception:
        # declutter is best-effort; never break the UI
        pass

def start_basic(n: int = 3):
    _g.startBasicGraph(int(max(3, min(10, n))))
    return json.dumps(_state_dict())

def add_random():
    ok, idx = _g.addRandomVertex()
    error_reason = None
    
    if not ok:
        # Determine the specific reason for failure
        if getattr(_g, "finalize_mode", False) or getattr(_g, "finalizeMode", False):
            error_reason = "Cannot add vertices in finalize mode"
        elif _g.periphery.size() < 2:
            error_reason = "Need at least 2 vertices on periphery"
        else:
            error_reason = "No valid periphery arc found (degree ≥ 5 rule for hidden vertices)"
    
    if ok:
        _post_layout_clean(1.15)  # remove overlaps after each insertion

    info = _g.get_last_add_info() or None
    info_out = None
    if info:
        sp = info.get("spawn_pos")
        fp = info.get("final_pos")
        info_out = {
            "index": int(info.get("index", -1)),
            "spawn_pos": _pt(sp) if sp else None,
            "final_pos": _pt(fp) if fp else None,
        }
    
    result = {"ok": bool(ok), "index": idx, "state": _state_dict(), "info": info_out}
    if error_reason:
        result["error_reason"] = error_reason
    
    return json.dumps(result)

def add_by_selection(a: int, b: int):
    ok, idx = _g.addVertexBySelection(int(a), int(b))
    error_reason = None
    
    if not ok:
        # Determine the specific reason for failure
        if getattr(_g, "finalize_mode", False) or getattr(_g, "finalizeMode", False):
            error_reason = "Cannot add vertices in finalize mode"
        elif _g.periphery.size() < 2:
            error_reason = "Need at least 2 vertices on periphery"
        else:
            error_reason = "Invalid selection or no valid arc between selected vertices"
    
    if ok:
        _post_layout_clean(1.15)

    info = _g.get_last_add_info() or None
    info_out = None
    if info:
        sp = info.get("spawn_pos")
        fp = info.get("final_pos")
        info_out = {
            "index": int(info.get("index", -1)),
            "spawn_pos": _pt(sp) if sp else None,
            "final_pos": _pt(fp) if fp else None,
        }
    
    result = {"ok": bool(ok), "index": idx, "state": _state_dict(), "info": info_out}
    if error_reason:
        result["error_reason"] = error_reason
    
    return json.dumps(result)

def redraw():
    _g.redraw_planar(iterations=None, radius=None, light=False)
    _post_layout_clean(1.05)  # tiny polish after redraw
    return json.dumps(_state_dict())

def get_state():
    return json.dumps(_state_dict())

def save_json_string():
    data = {
        "version": 1,
        "vertices": [
            {
                "index": v.getIndex(),
                "x": float(v.getPosition().x()),
                "y": float(v.getPosition().y()),
                "color": int(v.getColorIndex()),
                "visible": bool(v.isVisible()),
            }
            for v in _g.getVertices() if v is not None
        ],
        "edges": [
            {
                "u": e.getStartVertex().getIndex(),
                "v": e.getEndVertex().getIndex(),
                "visible": bool(e.isVisible()),
            }
            for e in _g.getEdges()
        ],
        "periphery": list(_g.getPeriphery()),
        "meta": _g.get_stats(),
        "labelMode": int(_g.get_label_mode()),
    }
    return json.dumps(data)

def load_json_string(json_str: str):
    try:
        data = json.loads(json_str)
        _g.loadFromJsonObject(data)
        return json.dumps(_state_dict())
    except Exception as e:
        return json.dumps({"error": str(e)})

def go_to(m: int):
    _g.goToVertex(int(m))
    return json.dumps(_state_dict())

def set_auto_tutte(on: bool):
    _g.set_auto_tutte_reembed(bool(on))
    return json.dumps({"ok": True})

def set_finalize(on: bool):
    _g.set_finalize_mode(bool(on))
    return json.dumps({"ok": True})

def set_label_mode(mode: int):
    _g.set_label_mode(int(mode))
    return json.dumps({"ok": True})

def declutter(intensity: float = 1.0):
    try:
        changed = _g.declutter_for_view(float(intensity))
        return json.dumps({"changed": changed, "state": _state_dict()})
    except Exception:
        changed = False
        return json.dumps({"changed": changed, "state": _state_dict()})

# New advanced layout methods
def set_layout_config(config: dict):
    """Update layout configuration parameters"""
    global _layout_config
    _layout_config.update(config)
    
    # Apply configuration to graph if possible
    try:
        if hasattr(_g, 'set_layout_config'):
            _g.set_layout_config(config)
    except:
        pass
    
    return json.dumps({"ok": True, "config": _layout_config})

def get_layout_config():
    """Get current layout configuration"""
    return json.dumps(_layout_config)

def calculate_layout_quality():
    """Calculate and return layout quality metrics"""
    try:
        # Calculate edge length uniformity
        edges = _g.getEdges()
        if not edges:
            return json.dumps({"quality": 0, "metrics": {}})
        
        lengths = []
        for e in edges:
            v1 = e.getStartVertex()
            v2 = e.getEndVertex()
            if v1 and v2:
                p1 = v1.getPosition()
                p2 = v2.getPosition()
                length = ((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2) ** 0.5
                lengths.append(length)
        
        if not lengths:
            return json.dumps({"quality": 0, "metrics": {}})
        
        # Calculate variance and quality score
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Quality score: lower variance = higher quality (0-100)
        quality = max(0, 100 - min(100, variance / 10))
        
        metrics = {
            "avg_edge_length": avg_length,
            "edge_length_variance": variance,
            "total_edges": len(lengths),
            "quality_score": quality
        }
        
        return json.dumps({"quality": quality, "metrics": metrics})
        
    except Exception as e:
        return json.dumps({"quality": 0, "error": str(e)})

def recalculate_forces():
    """Recalculate layout forces with current configuration"""
    try:
        # This would integrate with the enhanced force calculation system
        # For now, just trigger a redraw
        _g.redraw_planar(iterations=None, radius=None, light=True)
        return json.dumps({"ok": True, "message": "Forces recalculated"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})
