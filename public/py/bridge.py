# public/py/bridge.py
import json
import math
import statistics

from graph import Graph
from qtcore_shim import QPointF  # noqa

_g = Graph()
_g.set_auto_tutte_reembed(True)  # use pure-Python Tutte (GS) in browser


def _pt(p):
    return {"x": float(p.x()), "y": float(p.y())}


def _state_dict():
    verts = []
    for v in _g.getVertices():
        if v is None:
            continue
        p = v.getPosition()
        verts.append(
            {
                "index": v.getIndex(),
                "x": float(p.x()),
                "y": float(p.y()),
                "color": int(v.getColorIndex()),
                "visible": bool(v.isVisible()),
            }
        )
    edges = [
        {
            "u": e.getStartVertex().getIndex(),
            "v": e.getEndVertex().getIndex(),
            "visible": bool(e.isVisible()),
        }
        for e in _g.getEdges()
    ]
    return {
        "vertices": verts,
        "edges": edges,
        "periphery": list(_g.getPeriphery()),
        "meta": _g.get_stats(),
        "labelMode": int(_g.get_label_mode()),
    }


# ------------- Even-density layout helpers (planarity-safe) -------------
def _to_xy(v):
    p = v.getPosition()
    return (float(p.x()), float(p.y()))


def _set_xy(v, x, y):
    v.setPosition(QPointF(float(x), float(y)))


def _build_adj_and_edges():
    adj = {}
    edges = []
    for e in _g.getEdges():
        u = e.getStartVertex().getIndex()
        v = e.getEndVertex().getIndex()
        if u == v:
            continue
        edges.append((u, v))
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj, edges


def _median_edge_len(pos, edges):
    L = []
    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        L.append(math.hypot(x2 - x1, y2 - y1))
    if not L:
        return 1.0
    return statistics.median(L)


def _barycentric_relax(pos, adj, fixed, iters=150, omega=1.0):
    # Positive weights; Gauss–Seidel-style in-place updates.
    for _ in range(max(1, int(iters))):
        moved = 0.0
        for i, neigh in adj.items():
            if i in fixed or not neigh:
                continue
            sx = sy = 0.0
            cnt = 0
            for j in neigh:
                if j in pos:
                    x, y = pos[j]
                    sx += x
                    sy += y
                    cnt += 1
            if cnt == 0:
                continue
            nx, ny = sx / cnt, sy / cnt
            ox, oy = pos[i]
            nx = ox + omega * (nx - ox)
            ny = oy + omega * (ny - oy)
            pos[i] = (nx, ny)
            moved = max(moved, abs(nx - ox) + abs(ny - oy))
        if moved < 1e-7:
            break


def _edge_equalize(
    pos,
    edges,
    fixed,
    target,
    push_short=0.33,
    pull_long=0.20,
    low=0.72,
    high=1.60,
):
    # Gentle equalization toward target edge length
    t_lo = low * target
    t_hi = high * target
    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) + 1e-12
        ux, uy = dx / d, dy / d
        if d < t_lo:
            delta = push_short * (t_lo - d)
            if u not in fixed:
                pos[u] = (x1 - delta * ux, y1 - delta * uy)
            if v not in fixed:
                pos[v] = (x2 + delta * ux, y2 + delta * uy)
        elif d > t_hi:
            delta = pull_long * (d - t_hi)
            if u not in fixed:
                pos[u] = (x1 + delta * ux, y1 + delta * uy)
            if v not in fixed:
                pos[v] = (x2 - delta * ux, y2 - delta * uy)


def _even_density_layout(intensity=1.0, reproject_periphery=False):
    """
    Even-density planar layout (planarity-safe):
    - keep periphery fixed (optional uniform-circle re-projection),
    - Tutte-style barycentric relaxation,
    - edge-length equalization + short re-relax per round.
    """
    verts = [v for v in _g.getVertices() if v is not None and v.isVisible()]
    if len(verts) < 3:
        return False

    pos = {v.getIndex(): _to_xy(v) for v in verts}
    adj, edges = _build_adj_and_edges()
    per = [i for i in _g.getPeriphery() if i in pos]
    fixed = set(per)

    # Optional: reproject periphery onto a uniform circle (for maximum uniformity)
    if reproject_periphery and len(per) >= 3:
        cx = sum(pos[i][0] for i in per) / len(per)
        cy = sum(pos[i][1] for i in per) / len(per)
        # Orientation of current periphery
        area2 = 0.0
        for k in range(len(per)):
            x1, y1 = pos[per[k]][0] - cx, pos[per[k]][1] - cy
            x2, y2 = pos[per[(k + 1) % len(per)]][0] - cx, pos[per[(k + 1) % len(per)]][1] - cy
            area2 += x1 * y2 - x2 * y1
        orient = 1.0 if area2 > 0 else -1.0
        # Radius: preserve scale by mean radius
        r = sum(math.hypot(pos[i][0] - cx, pos[i][1] - cy) for i in per) / max(1, len(per))
        if r <= 1e-6:
            r = 100.0
        # Keep first vertex angle to reduce visual snap
        x0, y0 = pos[per[0]][0] - cx, pos[per[0]][1] - cy
        phi0 = math.atan2(y0, x0)
        step = 2 * math.pi / len(per)
        for k, i in enumerate(per):
            phi = phi0 + orient * k * step
            pos[i] = (cx + r * math.cos(phi), cy + r * math.sin(phi))

    # Main relax + equalize cycles
    total_rounds = max(1, int(3 + 3 * intensity))
    base_iters = max(80, int(120 * intensity))
    for _ in range(total_rounds):
        _barycentric_relax(pos, adj, fixed, iters=base_iters, omega=1.0)
        target = _median_edge_len(pos, edges)
        _edge_equalize(pos, edges, fixed, target, push_short=0.33, pull_long=0.20)
        _barycentric_relax(pos, adj, fixed, iters=30, omega=1.0)

    # Write back
    for v in verts:
        i = v.getIndex()
        if i in pos:
            x, y = pos[i]
            _set_xy(v, x, y)
    return True


def _post_layout_clean(intensity: float = 1.15):
    """
    Gentle, planarity-safe spacing tuned for interactive insertions.
    """
    try:
        # keep periphery as-is after local edits to avoid big snaps
        _even_density_layout(intensity=float(intensity), reproject_periphery=False)
    except Exception:
        pass
    # Optional: tiny polish from existing routine (best-effort, never fail)
    try:
        _g.declutter_for_view(1.02)
    except Exception:
        pass


# ------------- Commands exported to the worker -------------
def start_basic(n: int = 3):
    _g.startBasicGraph(int(max(3, min(10, n))))
    return json.dumps(_state_dict())


def add_random():
    ok, idx = _g.addRandomVertex()
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
    return json.dumps({"ok": bool(ok), "index": idx, "state": _state_dict(), "info": info_out})


def add_by_selection(a: int, b: int):
    ok, idx = _g.addVertexBySelection(int(a), int(b))
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
    return json.dumps({"ok": bool(ok), "index": idx, "state": _state_dict(), "info": info_out})


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
                "colorIndex": int(v.getColorIndex()),
                "origin": v.getOrigin(),
                "visible": bool(v.isVisible()),
            }
            for v in _g.getVertices()
            if v is not None
        ],
        "edges": [
            [e.getStartVertex().getIndex(), e.getEndVertex().getIndex()] for e in _g.getEdges()
        ],
        "periphery": list(_g.getPeriphery()),
        "labelMode": int(_g.get_label_mode()),
    }
    return json.dumps(data, indent=2)


def load_json_string(s: str):
    data = json.loads(s)
    _g.clear()

    from vertex import Vertex

    verts = data.get("vertices", [])
    if not isinstance(verts, list) or not verts:
        raise ValueError("JSON missing 'vertices' list.")
    max_idx = max(int(v["index"]) for v in verts)
    _g.vertices = [None] * (max_idx + 1)
    _g._adjacency = {}
    for rec in verts:
        idx = int(rec["index"])
        x = float(rec["x"])
        y = float(rec["y"])
        color_idx = int(rec.get("colorIndex", 1))
        origin = rec.get("origin", "manual")
        visible = bool(rec.get("visible", True))
        v = Vertex(idx, QPointF(x, y), colorIndex=color_idx, origin=origin)
        v.setVisible(visible)
        _g.vertices[idx] = v
        _g._adjacency[idx] = set()

    from edge import Edge

    edges_list = data.get("edges", [])
    if not isinstance(edges_list, list):
        edges_list = []
    for uv in edges_list:
        if not isinstance(uv, (list, tuple)) or len(uv) != 2:
            continue
        u, v = int(uv[0]), int(uv[1])
        if 0 <= u < len(_g.vertices) and 0 <= v < len(_g.vertices):
            if _g.vertices[u] is not None and _g.vertices[v] is not None and u != v:
                e = Edge(_g.vertices[u], _g.vertices[v])
                e.setVisible(_g.vertices[u].isVisible() and _g.vertices[v].isVisible())
                _g._add_edge_internal(e)

    peri = [
        int(i)
        for i in data.get("periphery", [])
        if 0 <= int(i) < len(_g.vertices) and _g.vertices[int(i)] is not None
    ]
    if len(peri) >= 3 and len(set(peri)) == len(peri):
        _g.periphery.initialize(peri)
    else:
        base = [i for i, v in enumerate(_g.vertices) if v is not None][:3]
        if len(base) == 3:
            _g.periphery.initialize(base)
        else:
            raise ValueError("Cannot rebuild periphery: not enough vertices.")

    try:
        _g.labelMode = int(data.get("labelMode", 2))
    except Exception:
        pass

    for e in _g.getEdges():
        u = e.getStartVertex().getIndex()
        v = e.getEndVertex().getIndex()
        e.setVisible(_g.vertices[u].isVisible() and _g.vertices[v].isVisible())

    _g._target_len = _g._compute_target_edge_length()
    return json.dumps(_state_dict())


def go_to(m: int):
    _g.goToVertex(int(m))
    return json.dumps(_state_dict())


def set_auto_tutte(on: bool):
    _g.set_auto_tutte_reembed(bool(on))
    return json.dumps({"ok": True})


def set_finalize(on: bool):
    # This flag is checked in addRandomVertex/addVertexBySelection
    _g.finalize_mode = bool(on)
    return json.dumps({"ok": True})


def set_label_mode(mode: int):
    try:
        _g.labelMode = int(mode)
    except Exception:
        pass
    return json.dumps({"labelMode": int(_g.labelMode)})


def declutter(intensity: float = 1.0):
    changed = False
    try:
        # For explicit declutter, we allow (slight) boundary reprojection for a very uniform look.
        # Set to False if you prefer to keep the current outer face shape.
        changed = bool(_even_density_layout(float(intensity), reproject_periphery=True))
    except Exception:
        changed = False
    # Optional: keep existing declutter as a light post-step
    try:
        _g.declutter_for_view(0.98 + 0.04 * float(intensity))
    except Exception:
        pass
    return json.dumps({"changed": changed, "state": _state_dict()})