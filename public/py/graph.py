# graph.py

from vertex import Vertex
from edge import Edge
from periphery import Periphery
from utils_geom import (
    v_add, v_sub, v_scale, v_len, v_norm, v_dot, v_rot90_ccw, v_rot90_cw
)
from qtcore_shim import QPointF
from typing import Optional
import math
import json
import secrets
import random
from statistics import median
from bisect import bisect_right
from dataclasses import dataclass, field

INITIAL_RADIUS = 200.0
PI = math.pi
    
@dataclass
class RedrawConfig:
    # Force-directed parameters
    spring_k: float = 0.18
    max_step_frac: float = 0.12
    repulsion_cutoff_frac: float = 2.8
    repulsion_strength_frac: float = 0.25
    
    # Congestion parameters
    congestion_cutoff_frac: float = 1.9
    congestion_beta_frac: float = 0.30
    congestion_thresh: int = 10
    
    # Tutte relaxation parameters
    tutte_rounds: int = 12
    adaptive_tutte_gamma: float = 1.30
    
    # Heuristic parameters
    long_edge_reduce_rounds: int = 2
    short_edge_boost_rounds: int = 2
    fan_spread_iters: int = 2
    
    # Final separation
    separation_iters: int = 2

@dataclass
class GraphConfig:
    # Define different configs for different graph sizes
    small_graph_v_thresh: int = 1200
    medium_graph_v_thresh: int = 6000

    # Default configs
    default: RedrawConfig = field(default_factory=RedrawConfig)
    
    # A lighter config for faster, less precise redraws
    light: RedrawConfig = field(default_factory=lambda: RedrawConfig(
        spring_k=0.16, max_step_frac=0.11, repulsion_cutoff_frac=2.4,
        repulsion_strength_frac=0.20, congestion_thresh=8,
        tutte_rounds=5, adaptive_tutte_gamma=1.20, long_edge_reduce_rounds=1,
        short_edge_boost_rounds=2, fan_spread_iters=1, separation_iters=1
    ))

class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.periphery = Periphery()
        self.labelMode = 2  # 0=color-only, 1=index-only, 2=color+index
        self._last_add_info = None
        self._adjacency = {}
        self.config = GraphConfig()
        

        # Robust randomness
        self._rng = random.Random(secrets.randbits(64))

        # Target edge length (EMA) to keep edge lengths homogeneous at scale
        self._target_len = None

        # Redraw & scale management
        self._adds_since_redraw = 0
        self._MAX_WORKING_RADIUS: Optional[float] = 150000.0

        # Auto-expand/view integration flags (UI reads these)
        self.auto_expand = True
        self.auto_expand_mode = "fit"  # "fit" | "infinite"
        self.view_padding = 40.0
        self.target_edge_px = 18.0

        # UI callback hook
        self.on_layout_changed = None

        # Debug info for random choice
        self.last_random_choice = None

        # Strict homogenization defaults (slightly tightened)
        self.strict_homog_enabled = True
        self.homog_tol = 9.0e-4            # relative tolerance on edge length (tighter)
        self.homog_max_rounds_small = 80   # V <= 1200
        self.homog_max_rounds_medium = 45  # V <= 5000
        self.homog_max_rounds_large = 20   # larger
        self.homog_clamp_frac = 0.10       # fraction of target length per move clamp

        # Ultra-homogeneity controls
        self.lock_target_len = True         # keep target length stable (based on periphery)
        self.ultra_homog = True             # enable ultra polishing
        self.ultra_small_threshold = 4000
        self.ultra_tol_small = 4.0e-4       # tighter tolerance for small graphs
        self.ultra_tol_large = 7.0e-4       # slightly looser for big graphs
        self.ultra_rounds_small = 150
        self.ultra_rounds_large = 90
        self.ultra_edge_proj_frac_small = 0.18  # project top 18% worst edges (small graphs)
        self.ultra_edge_proj_frac_large = 0.10  # project top 10% (large graphs)
        self.ultra_clamp_frac = 0.16            # allow stronger per-edge projection

        # Auto-polish after each add
        self.auto_polish_after_add = True
        self.auto_polish_p95 = 1.05         # stop when 95% of edges within ±5%
        self.auto_polish_max_passes = 2     # at most 2 extra passes per insertion
        self.always_tutte_reembed = True  # UI-controlled: auto Tutte on each add (default OFF)
        
    # --------------------------
    # Small helpers
    # --------------------------
    def set_auto_tutte_reembed(self, enabled: bool):
        """Enable/disable Tutte re-embed automatically on each vertex insertion."""
        self.always_tutte_reembed = bool(enabled)
    
    def _hash_u32(self, *args):
        h = 0x811C9DC5
        for a in args:
            x = int(a) & 0xffffffff
            h ^= x
            h = (h * 0x01000193) & 0xffffffff
        return h

    def _rotate_vec(self, v: QPointF, ang_rad: float) -> QPointF:
        c = math.cos(ang_rad)
        s = math.sin(ang_rad)
        return QPointF(v.x() * c - v.y() * s, v.x() * s + v.y() * c)

    def _approx_diameter(self, idx: int) -> float:
        if idx >= 1000:
            return 40.0
        elif idx >= 100:
            return 36.0
        elif idx >= 10:
            return 32.0
        else:
            return 30.0

    def _min_sep_length(self):
        vis = [v.getDiameter() for v in self.vertices if v and v.isVisible()]
        avg_d = (sum(vis) / len(vis)) if vis else 30.0
        tl = self._quick_target_length()
        return max(0.24 * tl, 0.55 * avg_d) + 3.0

    # --------------------------
    # Congestion detection and global scale-out
    # --------------------------
    def _estimate_min_vertex_spacing(self, dmin_hint: float) -> float:
        """Approximate minimum center-to-center spacing between visible vertices.
        Uses a grid hash (O(N)) to avoid O(N^2) pair checks, suitable for 10k+ vertices.
        """
        cell = max(1.0, float(dmin_hint))
        inv = 1.0 / cell
        grid = {}
        pts = []
        for i, v in enumerate(self.vertices):
            if v is None or not v.isVisible():
                continue
            p = v.getPosition()
            cx = int(math.floor(p.x() * inv)); cy = int(math.floor(p.y() * inv))
            grid.setdefault((cx, cy), []).append((i, p))
            pts.append((i, p, cx, cy))
        if len(pts) < 2:
            return float('inf')

        min_d2 = float('inf')
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
        seen = set()
        for i, p, cx, cy in pts:
            for dx, dy in neigh:
                for j, q in grid.get((cx+dx, cy+dy), []):
                    if j <= i:  # avoid double
                        continue
                    key = (i, j)
                    if key in seen:
                        continue
                    seen.add(key)
                    ddx = q.x() - p.x(); ddy = q.y() - p.y()
                    d2 = ddx*ddx + ddy*ddy
                    if d2 < min_d2:
                        min_d2 = d2
        return math.sqrt(min_d2) if min_d2 != float('inf') else float('inf')

    def _global_expand_if_congested(self, inflate_pad: float = 1.04) -> bool:
        """Uniformly scale out the layout if any vertex-vertex spacing is below dmin.
        Uniform scaling preserves planarity and improves edge-length homogeneity headroom.
        Returns True if scaling was applied.
        """
        dmin = self._min_sep_length()
        m = self._estimate_min_vertex_spacing(dmin)
        if not (m < dmin):
            return False
        # Scale factor to clear the tightest gap with a small pad
        s = max(1.0, float(dmin) / max(1e-9, float(m))) * float(inflate_pad)
        c = self._periphery_center()
        for v in self.vertices:
            p = v.getPosition()
            nx = c.x() + (p.x() - c.x()) * s
            ny = c.y() + (p.y() - c.y()) * s
            v.setPosition(QPointF(nx, ny))
        if self._target_len is not None:
            self._target_len *= s
        # Clamp if we exceed working radius
        self._normalize_and_recenter(max_radius=self._MAX_WORKING_RADIUS)
        return True

    # --------------------------
    # Segment intersection helpers
    # --------------------------
    def _post_add_settle(self, new_idx: int):
        """
        Lightweight, planarity-safe local settle used when we don't do a full Tutte re-embed.
        - Normalize star edge lengths around the new vertex.
        - Apply local vertex-vertex separation and vertex-edge clearance.
        - Crossing guard; fallback to Tutte re-embed if necessary.
        """
        per = list(self.periphery.getIndices())
        if len(per) < 3 or new_idx < 0 or new_idx >= len(self.vertices):
            return

        pinned = set(per)
        T = self._quick_target_length()
        if T <= 1e-9:
            T = self._compute_target_edge_length()
            self._target_len = T

        # Local neighborhood = new vertex + immediate neighbors
        neigh = list(self._adjacency.get(new_idx, set()))
        local_idxs = [new_idx] + neigh

        snap = self._snapshot_positions()

        try:
            # 1) Normalize edge lengths around the new vertex (center-only, with guard)
            self.enforce_unit_lengths_local(
                center_idx=new_idx, pinned=pinned, target_length=T,
                rounds=28, clamp_frac=0.10, tol=9e-4
            )

            # 2) Local vertex-vertex separation (keep it tight and local)
            self._separate_min_dist_grid(
                pinned=pinned, dmin=self._min_sep_length(),
                iters=2, clamp_step=0.10 * T, only_indices=local_idxs
            )

            # 3) Local vertex-edge clearance (avoid the new star hugging edges)
            self._ve_clearance_grid(
                pinned=pinned, clear_dist=0.28 * T,
                iters=2, clamp_step=0.10 * T, only_indices=local_idxs, skip_incident=True
            )

            # Optional: gentle fan spread if the new vertex has many neighbors
            if len(neigh) >= 4:
                self._spread_fans(pinned=pinned, angle_min_deg=9.0, iters=1, k_frac=0.05)

            # Crossing guard: if crossings remain, revert and escalate to Tutte
            if self._has_crossings_quick(cell_size=T, max_reports=1):
                self._restore_positions(snap)
                # Emergency fallback: guaranteed-planar layout (light polish)
                n_interior = max(0, len(self.vertices) - len(self.periphery.getIndices()))
                try:
                    self.reembed_tutte(use_scipy=(n_interior > 150), final_polish=True)
                except Exception:
                    self.reembed_tutte(use_scipy=False, final_polish=True)
                return

            # Refresh target length from current layout (keeps median edge close to T)
            self._target_len = self._compute_target_edge_length()

        finally:
            if self.on_layout_changed:
                self.on_layout_changed(self.get_bounding_box(), self._periphery_center(), self.target_edge_px)
                
                
    def _orient(self, a: QPointF, b: QPointF, c: QPointF):
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())

    def _on_segment(self, a: QPointF, b: QPointF, c: QPointF, eps=1e-9):
        return (min(a.x(), b.x()) - eps <= c.x() <= max(a.x(), b.x()) + eps and
                min(a.y(), b.y()) - eps <= c.y() <= max(a.y(), b.y()) + eps and
                abs(self._orient(a, b, c)) <= eps)

    def _segments_properly_intersect(self, a: QPointF, b: QPointF, c: QPointF, d: QPointF, eps=1e-9):
        o1 = self._orient(a, b, c)
        o2 = self._orient(a, b, d)
        o3 = self._orient(c, d, a)
        o4 = self._orient(c, d, b)
        if (o1 * o2 < -eps) and (o3 * o4 < -eps):
            return True
        # Treat touching/collinear as non-crossing (fine for planarity)
        if abs(o1) <= eps and self._on_segment(a, b, c, eps): return False
        if abs(o2) <= eps and self._on_segment(a, b, d, eps): return False
        if abs(o3) <= eps and self._on_segment(c, d, a, eps): return False
        if abs(o4) <= eps and self._on_segment(c, d, b, eps): return False
        return False

    def _bbox_disjoint(self, a, b, c, d, eps=1e-9):
        min_ax = min(a.x(), b.x()) - eps
        max_ax = max(a.x(), b.x()) + eps
        min_ay = min(a.y(), b.y()) - eps
        max_ay = max(a.y(), b.y()) + eps
        min_cx = min(c.x(), d.x()) - eps
        max_cx = max(c.x(), d.x()) + eps
        min_cy = min(c.y(), d.y()) - eps
        max_cy = max(c.y(), d.y()) + eps
        return (max_ax < min_cx) or (max_cx < min_ax) or (max_ay < min_cy) or (max_cy < min_ay)

    def _point_segment_distance(self, p: QPointF, a: QPointF, b: QPointF) -> float:
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = p.x(), p.y()
        abx, aby = (bx - ax), (by - ay)
        denom = abx * abx + aby * aby
        if denom <= 1e-18:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * abx + (py - ay) * aby) / denom
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        qx = ax + t * abx
        qy = ay + t * aby
        return math.hypot(px - qx, py - qy)

    def _new_edges_too_close_to_vertices(self, new_pos: QPointF, arc_indices) -> bool:
        base = max(6.0, 0.14 * self._quick_target_length())
        per_set = set(arc_indices)
        for vi in arc_indices:
            p2 = self.vertices[vi].getPosition()
            minx = min(new_pos.x(), p2.x()) - base
            maxx = max(new_pos.x(), p2.x()) + base
            miny = min(new_pos.y(), p2.y()) - base
            maxy = max(new_pos.y(), p2.y()) + base
            for vtx in self.vertices:
                if not vtx:
                    continue
                j = vtx.getIndex()
                if j in per_set:
                    continue
                pj = vtx.getPosition()
                if pj.x() < minx or pj.x() > maxx or pj.y() < miny or pj.y() > maxy:
                    continue
                d_need = 0.5 * (self._approx_diameter(len(self.vertices)) + vtx.getDiameter()) + base
                if self._point_segment_distance(pj, new_pos, p2) < d_need:
                    return True
        return False

    def _new_edges_cross_any(self, new_pos: QPointF, arc_indices):
        for vi in arc_indices:
            p2 = self.vertices[vi].getPosition()
            for e in self.edges:
                u = e.getStartVertex().getIndex()
                v = e.getEndVertex().getIndex()
                if u == vi or v == vi:
                    continue
                p3 = self.vertices[u].getPosition()
                p4 = self.vertices[v].getPosition()
                if self._bbox_disjoint(new_pos, p2, p3, p4):
                    continue
                if self._segments_properly_intersect(new_pos, p2, p3, p4):
                    return True
        return False

    # --------------------------
    # Base graph ops
    # --------------------------
    def clear(self):
        self.vertices.clear()
        self.edges.clear()
        self.periphery.clear()
        self._adjacency.clear()
        self._target_len = None
        self._adds_since_redraw = 0
        self.last_random_choice = None
        self._last_add_info = None

    def _add_edge_internal(self, edge):
        v1_idx = edge.getStartVertex().getIndex()
        v2_idx = edge.getEndVertex().getIndex()
        if v2_idx in self._adjacency.get(v1_idx, set()):
            return
        self.edges.append(edge)
        self._adjacency.setdefault(v1_idx, set()).add(v2_idx)
        self._adjacency.setdefault(v2_idx, set()).add(v1_idx)

    def getDegree(self, vertex_index):
        return len(self._adjacency.get(vertex_index, set()))

    def startBasicGraph(self, n: int = 3):
        """
        Initialize a new graph with n seed vertices.
        - n>=3: convex n-gon with triangulation by fanning from V0.
        """
        self.clear()
        try:
            n = int(n)
        except Exception:
            n = 3
        n = max(3, min(10, n))

        # Place vertices on a circle
        for i in range(n):
            angle = 2 * PI * i / n - PI / 2
            pos = QPointF(INITIAL_RADIUS * math.cos(angle), INITIAL_RADIUS * math.sin(angle))
            v = Vertex(i, pos, (i % 4) + 1, origin="seed")
            self.vertices.append(v)
            self._adjacency[i] = set()

        # Outer cycle edges
        for i in range(n):
            self._add_edge_internal(Edge(self.vertices[i], self.vertices[(i + 1) % n]))

        # Fan triangulation from V0: connect 0->i for i=2..n-2
        if n > 3:
            for i in range(2, n - 1):
                self._add_edge_internal(Edge(self.vertices[0], self.vertices[i]))

        # Enforce stored CW order for the periphery (Periphery expects CW-only)
        indices = list(range(n))
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            pi = self.vertices[indices[i]].getPosition()
            pj = self.vertices[indices[j]].getPosition()
            area += pi.x() * pj.y() - pj.x() * pi.y()
        # If area > 0, index order is CCW ⇒ reverse to make stored order CW
        if area > 0.0:
            indices.reverse()

        # Set periphery and visibility
        self.periphery.initialize(indices)
        self.goToVertex(n)

        # Initialize target length from periphery edges
        self._target_len = self._compute_target_edge_length()

    # --------------------------
    # Periphery geometry helpers
    # --------------------------
    def tutte_embed(self, radius=None, center=None, use_scipy=True,
                max_iter=3000, tol=1e-6, write_back=True):
        """
        Exact/high-accuracy Tutte embedding:
        - Pins current periphery on a convex circle in the stored (CW) order.
        - Solves the barycentric system for interior vertices (SciPy if available; GS fallback).
        - If write_back=True, writes coordinates to vertices, else returns list[QPointF].
        """
        per = list(self.periphery.getIndices())
        n = len(self.vertices)
        if n == 0 or len(per) < 3:
            return None

        # Center for the boundary
        if center is None:
            center = self._periphery_center()

        # Boundary radius
        if radius is None:
            # Choose radius from target length or existing radii
            T = self._quick_target_length()
            m = len(per)
            radius = max(200.0, (T or 60.0) / max(1e-12, 2.0 * math.sin(math.pi / max(3, m))))
        else:
            radius = float(radius)

        # 1) Boundary positions: stored order is CW; place them around a circle in CW order
        boundary_pos = {}
        m = len(per)
        for k, vi in enumerate(per):
            # CW angles: decrease angle around circle
            theta = - (2.0 * math.pi * k / m) - (math.pi / 2.0)
            x = center.x() + radius * math.cos(theta)
            y = center.y() + radius * math.sin(theta)
            boundary_pos[vi] = QPointF(x, y)

        # 2) Interior set and adjacency
        per_set = set(per)
        interior = [i for i in range(n) if i not in per_set and self.vertices[i] is not None]
        if not interior:
            if write_back:
                for vi, bp in boundary_pos.items():
                    self.vertices[vi].setPosition(bp)
                return None
            else:
                coords = [self.vertices[i].getPosition() for i in range(n)]
                for vi, bp in boundary_pos.items():
                    coords[vi] = bp
                return coords

        nbrs = self._adjacency
        index_interior = {v: idx for idx, v in enumerate(interior)}
        N = len(interior)

        # Boundary contribution for RHS
        bx_vals = [0.0] * N
        by_vals = [0.0] * N

        if use_scipy:
            try:
                import numpy as _np
                import scipy.sparse as _sp
                import scipy.sparse.linalg as _spla

                rows, cols, data = [], [], []
                for ii, v in enumerate(interior):
                    deg = 0.0
                    for u in nbrs.get(v, set()):
                        if u == v:
                            continue
                        w = 1.0  # unit weights (Tutte)
                        if u in index_interior:
                            jj = index_interior[u]
                            rows.append(ii); cols.append(jj); data.append(-w)
                        else:
                            p = boundary_pos[u]
                            bx_vals[ii] += w * p.x()
                            by_vals[ii] += w * p.y()
                        deg += w
                    rows.append(ii); cols.append(ii); data.append(deg)

                L = _sp.csr_matrix((data, (rows, cols)), shape=(N, N))
                bx = _np.array(bx_vals, dtype=float)
                by = _np.array(by_vals, dtype=float)
                X = _spla.spsolve(L, bx)
                Y = _spla.spsolve(L, by)

                if write_back:
                    # Write boundary
                    for vi, bp in boundary_pos.items():
                        self.vertices[vi].setPosition(bp)
                    # Write interior
                    for v, ii in index_interior.items():
                        self.vertices[v].setPosition(QPointF(float(X[ii]), float(Y[ii])))
                    return None
                else:
                    coords = [self.vertices[i].getPosition() for i in range(n)]
                    for vi, bp in boundary_pos.items():
                        coords[vi] = bp
                    for v, ii in index_interior.items():
                        coords[v] = QPointF(float(X[ii]), float(Y[ii]))
                    return coords

            except Exception as _exc:
                # Fallback to GS below
                print(f"[tutte_embed] SciPy unavailable/failed: {_exc}; falling back to GS.")

        # 3) Gauss–Seidel fallback
        pos = [self.vertices[i].getPosition() for i in range(n)]
        for vi, bp in boundary_pos.items():
            pos[vi] = bp

        lam = 1.0
        for _ in range(max(1, int(max_iter))):
            max_delta = 0.0
            for v in interior:
                neigh = nbrs.get(v, set())
                if not neigh:
                    continue
                sx = sy = 0.0
                deg = 0
                for u in neigh:
                    p = pos[u] if u in interior else boundary_pos[u]
                    sx += p.x(); sy += p.y(); deg += 1
                if deg == 0:
                    continue
                newx = sx / deg
                newy = sy / deg
                old = pos[v]
                nx = (1 - lam) * old.x() + lam * newx
                ny = (1 - lam) * old.y() + lam * newy
                dx = nx - old.x(); dy = ny - old.y()
                max_delta = max(max_delta, abs(dx), abs(dy))
                pos[v] = QPointF(nx, ny)
            if max_delta <= tol:
                break

        if write_back:
            for vi, bp in boundary_pos.items():
                self.vertices[vi].setPosition(bp)
            for v in interior:
                self.vertices[v].setPosition(pos[v])
            return None
        else:
            coords = [self.vertices[i].getPosition() for i in range(n)]
            for vi, bp in boundary_pos.items():
                coords[vi] = bp
            for v in interior:
                coords[v] = pos[v]
            return coords
        
    def reembed_tutte(self, radius=None, use_scipy=True, final_polish=True):
        """
        Recompute a guaranteed planar straight-line embedding (Tutte) with pinned convex boundary.
        Optionally do a small positive-weight polish (planarity-safe).
        """
        per = list(self.periphery.getIndices())
        if len(per) < 3:
            return
        center = self._periphery_center()

        # Exact barycentric embedding with convex boundary
        self.tutte_embed(
            radius=radius, center=center,
            use_scipy=use_scipy, max_iter=2000, tol=1e-6, write_back=True
        )

        if final_polish:
            # A couple of positive-weight smoothing rounds (planarity-safe)
            pinned = set(per)
            T = self._quick_target_length()
            for _ in range(2):
                self._tutte_relax_adaptive(
                    pinned=pinned, target_len=T, rounds=6,
                    gamma=1.25, wmin=0.40, wmax=3.0, gauss_seidel=True
                )

        # Normalize
        self._normalize_and_recenter(max_radius=self._MAX_WORKING_RADIUS)

        # Recompute target length from the new layout (always do this)
        self._target_len = self._compute_target_edge_length()

        # Notify UI if a callback is set
        if self.on_layout_changed:
            self.on_layout_changed(self.get_bounding_box(), self._periphery_center(), self.target_edge_px)
        
    def _periphery_pts(self):
        return [self.vertices[i].getPosition() for i in self.periphery.getIndices()]

    def _periphery_center(self):
        pts = self._periphery_pts()
        if not pts:
            return QPointF(0, 0)
        cx = sum(p.x() for p in pts) / len(pts)
        cy = sum(p.y() for p in pts) / len(pts)
        return QPointF(cx, cy)

    def _polygon_signed_area_pts(self, pts):
        if len(pts) < 3:
            return 0.0
        A = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            A += pts[i].x() * pts[j].y() - pts[j].x() * pts[i].y()
        return 0.5 * A

    def _periphery_orientation_ccw(self):
        area = self._polygon_signed_area_pts(self._periphery_pts())
        return 1 if area > 0 else (-1 if area < 0 else 0)

    def _point_in_polygon(self, pt, poly_pts):
        x, y = pt.x(), pt.y()
        inside = False
        n = len(poly_pts)
        if n < 3:
            return False
        for i in range(n):
            j = (i + 1) % n
            xi, yi = poly_pts[i].x(), poly_pts[i].y()
            xj, yj = poly_pts[j].x(), poly_pts[j].y()
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
        return inside

    # --------------------------
    # Target edge length control (periphery-first)
    # --------------------------
    def _compute_target_edge_length(self):
        # Prefer periphery edges (stable target)
        per = self.periphery.getIndices()
        if len(per) >= 3:
            pts = [self.vertices[i].getPosition() for i in per]
            lens = []
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                lens.append(v_len(v_sub(pts[j], pts[i])))
            if lens:
                return float(median(lens))

        # Fallback: median over visible edges
        lens2 = []
        for e in self.edges:
            if e.isVisible():
                p1 = e.getStartVertex().getPosition()
                p2 = e.getEndVertex().getPosition()
                lens2.append(v_len(v_sub(p2, p1)))
        return float(median(lens2)) if lens2 else 60.0

    def _quick_target_length(self):
        return self._target_len if self._target_len is not None else 60.0

    # --------------------------
    # Core add operations
    # --------------------------
    def _redraw_frequency(self):
        V = len(self.vertices)
        if V < 300: return 1
        if V < 1000: return 6
        if V < 3000: return 12
        return 25

    def addVertex(self, periphery_segment, origin="manual"):
        """
        Insert a new vertex outside the periphery, touching the CW arc defined by periphery_segment.
        Rules:
        - periphery_segment must be a CW-contiguous arc on the current periphery.
        - len(periphery_segment) >= 2.
        - Full-cycle arc:
            • Allowed for manual selection (any periphery size).
            • For non-manual (e.g., random), allowed only when periphery is a triangle.
        """
        # Validate arc contiguity and size
        if not self.periphery.isContiguous(periphery_segment) or len(periphery_segment) < 2:
            return False, -1

        per_sz = self.periphery.size()
        full_cycle = (len(periphery_segment) == per_sz)

        # Allow full-cycle for manual; otherwise only allow for triangle
        if full_cycle and origin != "manual" and per_sz != 3:
            return False, -1

        # Enforce exact CW arc Vp..Vq
        vp = periphery_segment[0]
        vq = periphery_segment[-1]
        expected = self.periphery.getSegment(vp, vq)
        if expected is None or list(expected) != list(periphery_segment):
            return False, -1

        # Create new vertex at a computed outward position
        newIdx = len(self.vertices)
        pos, outward_dir = self.computeNewVertexPosition(periphery_segment, return_outward=True)
        colorIdx = (newIdx % 4) + 1
        newVertex = Vertex(newIdx, pos, colorIdx, origin=origin)
        self.vertices.append(newVertex)
        self._adjacency[newIdx] = set()

        # Connect edges: center-out along the touched arc
        m = len(periphery_segment)
        center = (m - 1) * 0.5
        order = sorted(range(m), key=lambda i: abs(i - center))
        for k in order:
            idx = periphery_segment[k]
            self._add_edge_internal(Edge(newVertex, self.vertices[idx]))

        # Color and update periphery
        self._greedy_color(newIdx)
        if full_cycle:
            self.periphery.updateAfterFullCycle(vp, vq, newIdx)
        else:
            self.periphery.updateAfterAddition(periphery_segment, newIdx)

        # Optional re-embed (UI toggle)
        # Post-insert layout
        if self.always_tutte_reembed:
            try:
                n_interior = max(0, len(self.vertices) - len(self.periphery.getIndices()))
                self.reembed_tutte(use_scipy=(n_interior > 150), final_polish=True)
            except Exception:
                self.reembed_tutte(use_scipy=False, final_polish=True)
            try:
                if len(self.vertices) <= 4000:
                    pinned = set(self.periphery.getIndices())
                    self.polish_homogeneity_ultra(pinned, light=True)
            except Exception:
                pass
        else:
            # Local, safe settle for the new star (prevents overlaps when Tutte is OFF)
            self._post_add_settle(newIdx)

        # Animation spawn info
        final_pos = self.vertices[newIdx].getPosition()
        spawn_dist = max(1.5 * self._quick_target_length(), 40.0)
        mult = 2.6 if len(periphery_segment) <= 3 else 2.0
        if outward_dir is None or v_len(outward_dir) < 1e-6:
            radial = v_norm(v_sub(final_pos, self._periphery_center()))
            if v_len(radial) < 1e-6:
                radial = QPointF(1.0, 0.0)
            spawn_pos = v_add(final_pos, v_scale(radial, spawn_dist * mult))
        else:
            spawn_pos = v_add(final_pos, v_scale(outward_dir, spawn_dist * mult))

        self._last_add_info = {"index": newIdx, "spawn_pos": spawn_pos, "final_pos": final_pos}

        if self.on_layout_changed:
            self.on_layout_changed(self.get_bounding_box(), self._periphery_center(), self.target_edge_px)

        return True, newIdx

    # -------- Random helpers --------
    def _sample_uniform_valid_segment(self, peri, deg_min=5):
        n = len(peri)
        if n < 2:
            return None

        good = [1 if self.getDegree(peri[i]) >= deg_min else 0 for i in range(n)]

        if n == 2:
            s = self._rng.randint(0, 1)
            return s, 2

        if sum(good) == n:
            s = self._rng.randint(0, n - 1)
            L = self._rng.randint(2, n)
            return s, L

        good2 = good + good
        size2 = 2 * n
        next_bad = [0] * (size2 + 1)
        next_bad[size2] = size2
        for i in range(size2 - 1, -1, -1):
            next_bad[i] = i if good2[i] == 0 else next_bad[i + 1]

        pref = [0] * (n + 1)
        for s in range(n):
            start = s + 1
            nb = next_bad[start]
            span = nb - start
            if span > n - 2:
                span = n - 2
            if span < 0:
                span = 0
            pref[s + 1] = pref[s] + (span + 1)

        total = pref[n]
        if total <= 0:
            return None

        r = self._rng.randrange(total)
        s = bisect_right(pref, r) - 1
        r_local = r - pref[s]
        L = 2 + r_local
        return s, L

    def _sample_uniform_segment_all(self, peri, length_bias="uniform", gamma=1.35):
        n = len(peri)
        if n < 2:
            return None
        if n == 2:
            return self._rng.randint(0, 1), 2

        if length_bias == "uniform":
            L = 2 + self._rng.randrange(n - 1)
        else:
            if length_bias == "favor_long":
                weights = [float(L ** gamma) for L in range(2, n + 1)]
            elif length_bias == "favor_short":
                weights = [float((n + 2 - L) ** gamma) for L in range(2, n + 1)]
            else:
                weights = [1.0 for _ in range(2, n + 1)]
            total = sum(weights)
            r = self._rng.random() * total
            acc = 0.0
            L = 2
            for k, w in enumerate(weights, start=2):
                acc += w
                if r <= acc:
                    L = k
                    break

        s = self._rng.randrange(n)
        return s, L

    def _segment_respects_degree_safeguard(self, segment, deg_min_pre=4):
        """
        True if all interior vertices of segment have pre-add degree >= deg_min_pre.
        That ensures they will be >= deg_min_pre + 1 after the insertion.
        """
        if len(segment) < 2:
            return False
        for vi in segment[1:-1]:
            if self.getDegree(vi) < deg_min_pre:
                return False
        return True

    def addRandomVertex(self):
        """
        Add a vertex by picking two distinct periphery vertices uniformly at random
        and inserting on the CW arc from Vp to Vq (inclusive), per client spec.

        Steps:
        1) Get current periphery indices
        2) Choose Vp uniformly from periphery
        3) Choose Vq uniformly from periphery \ {Vp}
        4) Use CW segment(Vp, Vq) and add the vertex

        Finalize mode must be OFF to add vertices.

        Returns: (ok: bool, new_index: int)
        """
        # Safety guard: if a finalize mode flag is present and ON, block additions
        if getattr(self, "finalize_mode", False) or getattr(self, "finalizeMode", False):
            return False, -1

        n = self.periphery.size()
        if n < 2:
            return False, -1

        peri = list(self.periphery.getIndices())

        # Uniformly choose two distinct indices i != j in [0..n-1]
        i = self._rng.randrange(n)
        j = self._rng.randrange(n - 1)
        if j >= i:
            j += 1
        vp = peri[i]
        vq = peri[j]

        seg = self.periphery.getSegment(vp, vq)
        if seg is None or len(seg) < 2:
            return False, -1

        self.last_random_choice = {"start": vp, "end": vq, "length": len(seg)}
        return self.addVertex(seg, origin="random")

    def addVertexBySelection(self, a, b=None):
        """
        Manual selection:
        - If called with two endpoints (a, b), use the CW arc a..b inclusive.
        - If called with a full CW segment list, validate contiguity and exact arc.

        Differences from random add:
        - Full-cycle arcs are allowed for any periphery size.
        - No degree≥5 safeguard for interior (hidden) vertices.

        Finalize mode must be OFF to add vertices.
        """
        # Safety guard: if a finalize mode flag is present and ON, block additions
        if getattr(self, "finalize_mode", False) or getattr(self, "finalizeMode", False):
            return False, -1

        n = self.periphery.size()
        if n < 2:
            return False, -1

        # Build the CW segment
        if b is None:
            seg = list(a)
            if not seg or len(seg) < 2:
                return False, -1
            if not self.periphery.isContiguous(seg):
                return False, -1
            # Ensure exact CW arc Vp..Vq (no rotation mismatch)
            expected = self.periphery.getSegment(seg[0], seg[-1])
            if expected is None or list(expected) != seg:
                return False, -1
            # Note: full-cycle allowed (len(seg) may equal n)
        else:
            vp = int(a); vq = int(b)
            seg = self.periphery.getSegment(vp, vq)
            if seg is None or len(seg) < 2:
                return False, -1
            # Note: full-cycle allowed (len(seg) may equal n)

        # Manual selection: no degree≥5 safeguard
        return self.addVertex(seg, origin="manual")
    
    def goToVertex(self, m: int) -> None:
        """
        Show vertices with index < m and hide the rest.
        Edges are visible only if both endpoints are visible.

        Notes:
        - m is 1-based from the UI perspective (Gm command). Internally we compare
        vertex indices (0-based) with m, so indices 0..m-1 remain visible.
        - If m <= 0, hides all. If m > current vertex count, shows all.
        """
        try:
            m = int(m)
        except Exception:
            m = len(self.vertices)

        if m < 0:
            m = 0
        if m > len(self.vertices):
            m = len(self.vertices)

        # Update vertex visibility
        for v in self.vertices:
            if v is None:
                continue
            v.setVisible(v.getIndex() < m)

        # Update edge visibility (edge visible only if both endpoints are visible)
        for e in self.edges:
            u = e.getStartVertex()
            v = e.getEndVertex()
            e.setVisible(u.isVisible() and v.isVisible())

    # --------------------------
    # Placement and relaxation
    # --------------------------
    def _circle_two_points_outward(self, a: QPointF, b: QPointF, r: float, outward_dir: QPointF) -> Optional[QPointF]:
        dx = b.x() - a.x()
        dy = b.y() - a.y()
        d = math.hypot(dx, dy)
        if d < 1e-9 or d > 2.0 * r + 1e-9:
            return None

        mx = 0.5 * (a.x() + b.x())
        my = 0.5 * (a.y() + b.y())
        h2 = r * r - 0.25 * d * d
        if h2 < 0:
            h2 = 0.0
        h = math.sqrt(h2)

        ux = -dy / max(d, 1e-9)
        uy = dx / max(d, 1e-9)

        q1 = QPointF(mx + ux * h, my + uy * h)
        q2 = QPointF(mx - ux * h, my - uy * h)

        v1x, v1y = q1.x() - mx, q1.y() - my
        v2x, v2y = q2.x() - mx, q2.y() - my
        s1 = v1x * outward_dir.x() + v1y * outward_dir.y()
        s2 = v2x * outward_dir.x() + v2y * outward_dir.y()
        return q1 if s1 >= s2 else q2
    def _ve_clearance_grid(self, pinned, clear_dist, iters=2, clamp_step=None,
                        only_indices=None, skip_incident=True):
        """
        Push non-pinned vertices away from nearby edges (vertex–edge clearance).
        Grid-accelerated: indexes edge AABBs expanded by clear_dist.
        """
        if clear_dist <= 0:
            return
        if clamp_step is None:
            clamp_step = 0.28 * max(clear_dist, 1.0)

        # Helper: projection on segment (ax,ay)-(bx,by)
        def _proj(px, py, ax, ay, bx, by):
            abx = bx - ax; aby = by - ay
            denom = abx*abx + aby*aby
            if denom <= 1e-18:
                return ax, ay, 0.0
            t = ((px - ax)*abx + (py - ay)*aby) / denom
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            qx = ax + t * abx; qy = ay + t * aby
            return qx, qy, t

        # Build set of vertices to process
        if only_indices is None:
            idxs = [i for i in range(len(self.vertices)) if self.vertices[i].isVisible()]
        else:
            idxs = [i for i in only_indices if self.vertices[i].isVisible()]

        # Precompute edge list and edge grid
        edges = []
        for e in self.edges:
            if not e.isVisible():
                continue
            u = e.getStartVertex().getIndex()
            v = e.getEndVertex().getIndex()
            edges.append((u, v))

        if not edges or not idxs:
            return

        # Grid: cell size = clear_dist
        cell = clear_dist
        inv_cell = 1.0 / max(1e-9, cell)

        for _ in range(max(1, iters)):
            # Build grid of expanded edge AABBs
            grid = {}
            pts = [v.getPosition() for v in self.vertices]
            for ei, (u, v) in enumerate(edges):
                p1 = pts[u]; p2 = pts[v]
                minx = min(p1.x(), p2.x()) - clear_dist
                maxx = max(p1.x(), p2.x()) + clear_dist
                miny = min(p1.y(), p2.y()) - clear_dist
                maxy = max(p1.y(), p2.y()) + clear_dist
                cx0 = int(math.floor(minx * inv_cell)); cx1 = int(math.floor(maxx * inv_cell))
                cy0 = int(math.floor(miny * inv_cell)); cy1 = int(math.floor(maxy * inv_cell))
                for cx in range(cx0, cx1 + 1):
                    for cy in range(cy0, cy1 + 1):
                        grid.setdefault((cx, cy), []).append(ei)

            # Accumulate pushes per vertex
            moves = {}
            for i in idxs:
                if i in pinned:
                    continue
                pi = pts[i]; px, py = pi.x(), pi.y()
                cx = int(math.floor(px * inv_cell)); cy = int(math.floor(py * inv_cell))

                accx = accy = 0.0
                # Neighborhood cells
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        eids = grid.get((cx + dx, cy + dy), [])
                        for ei in eids:
                            u, v = edges[ei]
                            if skip_incident and (i == u or i == v):
                                continue
                            pA = pts[u]; pB = pts[v]
                            qx, qy, t = _proj(px, py, pA.x(), pA.y(), pB.x(), pB.y())
                            dx0 = px - qx; dy0 = py - qy
                            dist = math.hypot(dx0, dy0)
                            if dist >= clear_dist or dist < 1e-9:
                                continue
                            deficit = (clear_dist - dist)
                            nx = dx0 / (dist if dist > 1e-9 else 1.0)
                            ny = dy0 / (dist if dist > 1e-9 else 1.0)
                            # Stronger in segment interior; taper near endpoints to protect angles
                            w_end = 0.40 + 0.60 * min(t, 1 - t)
                            w = min(1.0, (deficit / clear_dist)) * w_end
                            fmag = deficit * (0.85 * w)
                            accx += nx * fmag; accy += ny * fmag

                if accx != 0.0 or accy != 0.0:
                    mv_len = math.hypot(accx, accy)
                    if mv_len > clamp_step:
                        s = clamp_step / mv_len
                        accx *= s; accy *= s
                    moves[i] = QPointF(accx, accy)

            # Apply moves (with a minimal planarity guard)
            for i, mv in moves.items():
                pi = self.vertices[i].getPosition()
                cand = QPointF(pi.x() + mv.x(), pi.y() + mv.y())
                # Tiny guard: if moving creates crossings with incident edges, reduce step
                # (cheap approximation: try half step)
                if self._has_crossings_quick(max_reports=1):
                    cand = QPointF(pi.x() + 0.5 * mv.x(), pi.y() + 0.5 * mv.y())
                self.vertices[i].setPosition(cand)

    def _sample_uniform_valid_segment_fast(self, peri, require_new_deg_ge5=True, deg_min_pre=4):
        """
        O(n) preprocessing + O(1) sampling of one valid segment.
        Valid = contiguous CW arc (length L) such that:
        - 2 <= L <= n-1  (no full-cycle arc)
        - all interior vertices have deg >= deg_min_pre (so they become >= deg_min_pre+1)
        - if require_new_deg_ge5: L >= 5 (so new vertex has deg >= 5)
        Returns (start_index_in_peri, L) or None.
        """
        n = len(peri)
        if n < 2:
            return None

        # Pre-degrees of periphery vertices
        deg_ok = [1 if self.getDegree(peri[i]) >= deg_min_pre else 0 for i in range(n)]
        # Quick edge case: n == 2 → only arc length 2 possible, but L must be <= n-1 (disallowed)
        if n == 2:
            return None

        # Build "next bad" over doubled array to get max span of consecutive good after s+1
        good2 = deg_ok + deg_ok
        size2 = 2 * n
        next_bad = [0] * (size2 + 1)
        next_bad[size2] = size2
        for i in range(size2 - 1, -1, -1):
            next_bad[i] = i if good2[i] == 0 else next_bad[i + 1]

        Lmin = 5 if require_new_deg_ge5 else 2  # min arc length (new vertex degree constraint)
        total = 0
        weights = [0] * n

        # Compute how many valid lengths per start s
        for s in range(n):
            start = s + 1                   # first interior position
            nb = next_bad[start]            # first bad index after start
            span = nb - start               # number of consecutive good interiors
            # Cap span so that L <= n-1 (no full-cycle)
            span = min(span, n - 3)         # because L = 2 + span, and we want L <= n-1
            low = max(Lmin, 2)
            high = min(n - 1, 2 + span)
            w = max(0, high - low + 1)
            weights[s] = w
            total += w

        if total <= 0:
            return None

        r = self._rng.randrange(total)
        s = 0
        while r >= weights[s]:
            r -= weights[s]
            s += 1

        # Weights[s] = number of allowable L values starting at s
        start = s + 1
        nb = next_bad[start]
        span = min(nb - start, n - 3)
        low = max(Lmin, 2)
        # r is an offset in [0, weights[s)-1]; map to L = low + r
        L = low + r
        return s, L

    def computeNewVertexPosition(self, peripheryIndices, return_outward=False):
        """
        Deterministic outward spawn for a new vertex that will be re-embedded by Tutte.
        We deliberately do NOT run local crossing/clearance searches here, because
        the Tutte re-embed (with convex pinned boundary) guarantees a crossing-free
        final embedding. This keeps adds fast and removes 'heuristics failed' warnings.
        """
        assert len(peripheryIndices) >= 2

        # 1) Arc centroid
        arc_pts = [self.vertices[i].getPosition() for i in peripheryIndices]
        arc_centroid = QPointF(
            sum(p.x() for p in arc_pts) / len(arc_pts),
            sum(p.y() for p in arc_pts) / len(arc_pts)
        )

        # 2) Base outward direction (radial from periphery center to arc centroid)
        per_center = self._periphery_center()
        radial_out = v_norm(v_sub(arc_centroid, per_center))
        if v_len(radial_out) < 1e-6:
            radial_out = QPointF(1.0, 0.0)

        # 3) Stabilize outward_dir using boundary normals of the arc (helps for skewed rims)
        normals = []
        orient_ccw = (self._periphery_orientation_ccw() > 0)
        for i in range(len(peripheryIndices) - 1):
            a = self.vertices[peripheryIndices[i]].getPosition()
            b = self.vertices[peripheryIndices[i + 1]].getPosition()
            e = v_sub(b, a); L = v_len(e)
            if L < 1e-6:
                continue
            n = v_rot90_cw(e) if orient_ccw else v_rot90_ccw(e)
            nL = v_len(n)
            if nL > 1e-6:
                normals.append(v_scale(n, 1.0 / nL))

        if normals:
            nx = sum(n.x() for n in normals) / len(normals)
            ny = sum(n.y() for n in normals) / len(normals)
            outward_dir = QPointF(nx, ny)
            if v_len(outward_dir) < 1e-6:
                outward_dir = radial_out
            else:
                outward_dir = v_scale(outward_dir, 1.0 / v_len(outward_dir))
                # Ensure we point roughly away from the center
                if v_dot(outward_dir, radial_out) < 0:
                    outward_dir = v_scale(outward_dir, -1.0)
        else:
            outward_dir = radial_out

        # 4) Choose a comfortable radius outside the current rim
        per_pts_all = self._periphery_pts()
        per_center_all = self._periphery_center()
        per_radii = [v_len(v_sub(p, per_center_all)) for p in per_pts_all] if per_pts_all else [INITIAL_RADIUS]
        rim_R = max(per_radii) if per_radii else INITIAL_RADIUS

        target = self._quick_target_length() or 60.0
        L_arc = len(peripheryIndices)

        # Margin: larger for shorter arcs (they protrude more), smaller for long arcs
        base_margin = max(20.0, 0.15 * target)
        if L_arc <= 2:
            margin = base_margin * 2.4
        elif L_arc == 3:
            margin = base_margin * 2.0
        elif L_arc == 4:
            margin = base_margin * 1.6
        elif L_arc <= 6:
            margin = base_margin * 1.3
        else:
            margin = base_margin

        # Final radius: rim + margin + a small target-based offset
        R = rim_R + margin + 0.8 * target

        cand = v_add(arc_centroid, v_scale(outward_dir, R))

        if return_outward:
            return cand, outward_dir
        return cand

    # --------------------------
    # Incremental local relax
    # --------------------------
    def _incremental_relax(self, seed_idx, local_rounds=3):
        per_set = set(self.periphery.getIndices())
        local = set([seed_idx])
        local.update(self._adjacency.get(seed_idx, set()))
        for u in list(local):
            local.update(self._adjacency.get(u, set()))

        for _ in range(max(1, local_rounds)):
            for i in list(local):
                if i in per_set or not self.vertices[i].isVisible():
                    continue
                neigh = self._adjacency.get(i, set())
                if not neigh:
                    continue
                sx = sy = 0.0
                cnt = 0
                for nb in neigh:
                    p = self.vertices[nb].getPosition()
                    sx += p.x(); sy += p.y(); cnt += 1
                if cnt > 0:
                    self.vertices[i].setPosition(QPointF(sx / cnt, sy / cnt))

    # --------------------------
    # Hard min-distance separation (grid accelerated)
    # --------------------------
    def _separate_min_dist_grid(self, pinned, dmin, iters=2, clamp_step=None, only_indices=None):
        if dmin <= 0: return
        if clamp_step is None: clamp_step = dmin * 0.6

        def build_index_set():
            if only_indices is None:
                return [i for i in range(len(self.vertices)) if self.vertices[i].isVisible()]
            return [i for i in only_indices if self.vertices[i].isVisible()]

        for _ in range(max(1, iters)):
            idxs = build_index_set()
            cell = dmin; inv_cell = 1.0 / max(1e-9, cell)
            grid = {}
            pos = {i: self.vertices[i].getPosition() for i in idxs}
            for i in idxs:
                p = pos[i]
                cx = int(math.floor(p.x() * inv_cell))
                cy = int(math.floor(p.y() * inv_cell))
                grid.setdefault((cx, cy), []).append(i)

            moves = {i: QPointF(0, 0) for i in idxs if i not in pinned}
            for i in idxs:
                pi = pos[i]
                cxi = int(math.floor(pi.x() * inv_cell))
                cyi = int(math.floor(pi.y() * inv_cell))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cell_list = grid.get((cxi + dx, cyi + dy), [])
                        for j in cell_list:
                            if j <= i: continue
                            pj = pos[j]
                            dxij = pj.x() - pi.x(); dyij = pj.y() - pi.y()
                            dist = math.hypot(dxij, dyij)
                            if dist < 1e-9 or dist >= dmin: continue
                            ov = (dmin - dist)
                            if dist < 1e-9:
                                ux, uy = 1.0, 0.0
                            else:
                                ux, uy = dxij / dist, dyij / dist
                            mi = ov * 0.5; mj = ov * 0.5
                            if i in pinned and j in pinned:
                                continue
                            elif i in pinned:
                                mi = 0.0; mj = ov
                            elif j in pinned:
                                mi = ov; mj = 0.0

                            if i not in pinned:
                                mv = v_scale(QPointF(-ux, -uy), min(mi, clamp_step))
                                moves[i] = v_add(moves.get(i, QPointF(0, 0)), mv)
                            if j not in pinned:
                                mv = v_scale(QPointF(ux, uy), min(mj, clamp_step))
                                moves[j] = v_add(moves.get(j, QPointF(0, 0)), mv)

            for i, mv in moves.items():
                if mv.x() == 0 and mv.y() == 0: continue
                p = self.vertices[i].getPosition()
                self.vertices[i].setPosition(v_add(p, mv))

    # --------------------------
    # Grid-accelerated repulsion (for redraw cycles)
    # --------------------------
    def _repulsion_grid(self, pinned, cutoff, strength):
        if cutoff <= 0 or strength <= 0: return {}

        cell = cutoff; inv_cell = 1.0 / max(1e-9, cell)
        grid = {}; pos_cache = {}
        N = len(self.vertices)
        for i in range(N):
            v = self.vertices[i]
            if not v.isVisible(): continue
            p = v.getPosition()
            pos_cache[i] = p
            cx = int(math.floor(p.x() * inv_cell))
            cy = int(math.floor(p.y() * inv_cell))
            grid.setdefault((cx, cy), []).append(i)

        forces = {}
        for i in range(N):
            if i in pinned: continue
            vi = self.vertices[i]
            if not vi.isVisible(): continue
            pi = pos_cache[i]
            cx = int(math.floor(pi.x() * inv_cell))
            cy = int(math.floor(pi.y() * inv_cell))
            neigh_cells = [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            accx = accy = 0.0
            for cc in neigh_cells:
                if cc not in grid: continue
                for j in grid[cc]:
                    if j == i: continue
                    if j in self._adjacency.get(i, set()): continue
                    pj = pos_cache[j]
                    dx = pi.x() - pj.x(); dy = pi.y() - pj.y()
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6 or dist > cutoff: continue
                    t = 1.0 - dist / cutoff
                    fmag = strength * (t * t)
                    accx += (dx / dist) * fmag
                    accy += (dy / dist) * fmag
            if accx != 0.0 or accy != 0.0:
                forces[i] = QPointF(accx, accy)
        return forces

    # --------------------------
    # Congestion-aware forces (declump dense patches)
    # --------------------------
    def _congestion_forces(self, pinned, cutoff, beta, thresh, target_len):
        if cutoff <= 0 or beta <= 0: return {}

        cell = cutoff; inv_cell = 1.0 / max(1e-9, cell)
        grid = {}; pos_cache = {}
        N = len(self.vertices)
        for i in range(N):
            v = self.vertices[i]
            if not v.isVisible(): continue
            p = v.getPosition()
            pos_cache[i] = p
            cx = int(math.floor(p.x() * inv_cell))
            cy = int(math.floor(p.y() * inv_cell))
            grid.setdefault((cx, cy), []).append(i)

        forces = {}
        for i in range(N):
            if i in pinned: continue
            if not self.vertices[i].isVisible(): continue
            pi = pos_cache[i]
            cx = int(math.floor(pi.x() * inv_cell))
            cy = int(math.floor(pi.y() * inv_cell))
            neigh_cells = [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

            cnt = 0; sx = sy = 0.0
            for cc in neigh_cells:
                if cc not in grid: continue
                for j in grid[cc]:
                    if j == i: continue
                    if j in self._adjacency.get(i, set()): continue
                    pj = pos_cache[j]
                    dx = pi.x() - pj.x(); dy = pi.y() - pj.y()
                    dist = math.hypot(dx, dy)
                    if dist <= cutoff:
                        cnt += 1; sx += pj.x(); sy += pj.y()

            if cnt >= thresh:
                cxn = sx / cnt; cyn = sy / cnt
                dx = pi.x() - cxn; dy = pi.y() - cyn
                d = math.hypot(dx, dy)
                if d > 1e-9:
                    scale = min(0.6, (cnt - thresh + 1) / (thresh))
                    fmag = beta * scale
                    fx = dx / d * fmag; fy = dy / d * fmag
                    forces[i] = v_add(forces.get(i, QPointF(0, 0)), QPointF(fx, fy))
        return forces

    # --------------------------
    # Long-edge reducer (aggressively contracts very long edges)
    # --------------------------
    def _long_edge_reduce(self, pinned, target_length, hi1=1.30, hi2=1.85,
                          gain1=0.55, gain2=0.75, max_step=None, rounds=1):
        if target_length <= 0:
            return
        if max_step is None:
            max_step = 0.16 * target_length

        for _ in range(max(1, rounds)):
            for e in self.edges:
                if not e.isVisible(): continue
                u = e.getStartVertex().getIndex()
                v = e.getEndVertex().getIndex()
                pu = self.vertices[u].getPosition()
                pv = self.vertices[v].getPosition()
                dx = pv.x() - pu.x(); dy = pv.y() - pu.y()
                dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    continue

                ratio = dist / target_length
                if ratio <= hi1:
                    continue

                # stronger as ratio approaches hi2
                t = (ratio - hi1) / max(1e-9, (hi2 - hi1))
                t = max(0.0, min(1.0, t))
                gain = (1 - t) * gain1 + t * gain2

                corr = (dist - target_length) * gain
                ux, uy = dx / dist, dy / dist

                # Move endpoints toward each other to shorten (or only free end if one is on periphery)
                if (u not in pinned) and (v not in pinned):
                    mv = min(max_step, 0.5 * corr)
                    self.vertices[u].setPosition(QPointF(pu.x() + ux * mv, pu.y() + uy * mv))
                    self.vertices[v].setPosition(QPointF(pv.x() - ux * mv, pv.y() - uy * mv))
                elif (u in pinned) and (v not in pinned):
                    mv = min(max_step, corr)
                    # small tangential nudge to avoid collapsing angles around pinned node
                    tx, ty = -uy, ux
                    self.vertices[v].setPosition(QPointF(pv.x() - ux * mv + 0.06 * tx * mv,
                                                         pv.y() - uy * mv + 0.06 * ty * mv))
                elif (v in pinned) and (u not in pinned):
                    mv = min(max_step, corr)
                    tx, ty = -uy, ux
                    self.vertices[u].setPosition(QPointF(pu.x() + ux * mv + 0.06 * tx * mv,
                                                         pu.y() + uy * mv + 0.06 * ty * mv))
                # both pinned: skip

    # --------------------------
    # Short-edge boost (opens tiny triangles)
    # --------------------------
    def _short_edge_boost(self, pinned, target_length, factor=0.34, thresh=0.66, max_step=None, rounds=2):
        if target_length <= 0: return
        if max_step is None: max_step = target_length * 0.12

        for _ in range(max(1, rounds)):
            forces = {}
            for e in self.edges:
                if not e.isVisible(): continue
                u = e.getStartVertex().getIndex()
                v = e.getEndVertex().getIndex()
                if u in pinned and v in pinned: continue
                pu = self.vertices[u].getPosition()
                pv = self.vertices[v].getPosition()
                diff = v_sub(pv, pu)
                dist = v_len(diff)
                if dist < 1e-9: continue
                if dist < thresh * target_length:
                    deficit = (thresh * target_length - dist)
                    fmag = factor * deficit
                    dir_uv = v_scale(diff, 1.0 / dist)
                    fvec = v_scale(dir_uv, fmag)
                    if u not in pinned:
                        forces[u] = v_sub(forces.get(u, QPointF(0, 0)), fvec)
                    if v not in pinned:
                        forces[v] = v_add(forces.get(v, QPointF(0, 0)), fvec)
            for i, f in forces.items():
                flen = v_len(f)
                if flen > max_step:
                    f = v_scale(f, max_step / max(1e-9, flen))
                pos = self.vertices[i].getPosition()
                self.vertices[i].setPosition(v_add(pos, f))

    # --------------------------
    # Edge-length homogenizer (soft)
    # --------------------------
    def _edge_length_homogenize(self, pinned, target_length,
                                lo=0.94, hi=1.06, gain=0.50,
                                max_step=None, rounds=2):
        if target_length <= 1e-9: return
        if max_step is None: max_step = target_length * 0.10

        for _ in range(max(1, rounds)):
            for e in self.edges:
                if not e.isVisible(): continue
                u = e.getStartVertex().getIndex()
                v = e.getEndVertex().getIndex()
                if u in pinned and v in pinned: continue

                pu = self.vertices[u].getPosition()
                pv = self.vertices[v].getPosition()
                dx = pv.x() - pu.x(); dy = pv.y() - pu.y()
                dist = math.hypot(dx, dy)
                if dist < 1e-9: continue

                ratio = dist / target_length
                if lo <= ratio <= hi: continue

                ux, uy = dx / dist, dy / dist
                corr = (target_length - dist) * gain

                if u in pinned and v not in pinned:
                    mvx, mvy = ux * corr, uy * corr
                    mv_len = math.hypot(mvx, mvy)
                    if mv_len > max_step:
                        s = max_step / mv_len
                        mvx, mvy = mvx * s, mvy * s
                    self.vertices[v].setPosition(QPointF(pv.x() + mvx, pv.y() + mvy))

                elif v in pinned and u not in pinned:
                    mvx, mvy = -ux * corr, -uy * corr
                    mv_len = math.hypot(mvx, mvy)
                    if mv_len > max_step:
                        s = max_step / mv_len
                        mvx, mvy = mvx * s, mvy * s
                    self.vertices[u].setPosition(QPointF(pu.x() + mvx, pu.y() + mvy))

                else:
                    mvux, mvuy = -ux * (0.5 * corr), -uy * (0.5 * corr)
                    mvvx, mvvy =  ux * (0.5 * corr),  uy * (0.5 * corr)
                    len_u = math.hypot(mvux, mvuy)
                    if len_u > max_step:
                        s = max_step / len_u
                        mvux, mvuy = mvux * s, mvuy * s
                    len_v = math.hypot(mvvx, mvvy)
                    if len_v > max_step:
                        s = max_step / len_v
                        mvvx, mvvy = mvvx * s, mvvy * s
                    self.vertices[u].setPosition(QPointF(pu.x() + mvux, pu.y() + mvuy))
                    self.vertices[v].setPosition(QPointF(pv.x() + mvvx, pv.y() + mvvy))

    # --------------------------
    # Strict unit-length projector (near-exact, with crossing guard)
    # --------------------------
    def _snapshot_positions(self):
        return [self.vertices[i].getPosition() for i in range(len(self.vertices))]

    def _restore_positions(self, snap):
        for i, p in enumerate(snap):
            self.vertices[i].setPosition(p)

    def _has_crossings_quick(self, cell_size=None, max_reports=1):
        edges = [(e.getStartVertex().getIndex(), e.getEndVertex().getIndex())
                 for e in self.edges if e.isVisible()]
        if len(edges) <= 1:
            return False

        if cell_size is None:
            cell_size = self._quick_target_length()

        pts = [v.getPosition() for v in self.vertices]
        grid = {}
        aabbs = []

        def put(aabb, idx):
            (minx, miny, maxx, maxy) = aabb
            cx0 = int(math.floor(minx / cell_size)); cx1 = int(math.floor(maxx / cell_size))
            cy0 = int(math.floor(miny / cell_size)); cy1 = int(math.floor(maxy / cell_size))
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    grid.setdefault((cx, cy), []).append(idx)

        for idx, (u, v) in enumerate(edges):
            p1, p2 = pts[u], pts[v]
            aabb = (min(p1.x(), p2.x()), min(p1.y(), p2.y()),
                    max(p1.x(), p2.x()), max(p1.y(), p2.y()))
            aabbs.append(aabb)
            put(aabb, idx)

        seen = set()
        hits = 0
        for cell, eidxs in grid.items():
            L = len(eidxs)
            if L < 2: continue
            for ii in range(L):
                i = eidxs[ii]
                u1, v1 = edges[i]
                a, b = pts[u1], pts[v1]
                for jj in range(ii + 1, L):
                    j = eidxs[jj]
                    key = (min(i, j), max(i, j))
                    if key in seen: continue
                    seen.add(key)
                    u2, v2 = edges[j]
                    if u1 in (u2, v2) or v1 in (u2, v2): continue
                    c, d = pts[u2], pts[v2]
                    A = aabbs[i]; B = aabbs[j]
                    if (A[2] < B[0]) or (B[2] < A[0]) or (A[3] < B[1]) or (B[3] < A[1]):
                        continue
                    if self._segments_properly_intersect(a, b, c, d):
                        hits += 1
                        if hits >= max_reports:
                            return True
        return False

    def enforce_unit_lengths_strict(self, pinned, target_length, tol=1.2e-3,
                                    max_rounds=60, clamp_frac=0.08, separation_iters=1):
        if target_length <= 1e-9 or not self.edges:
            return

        step = max(1e-6, clamp_frac * target_length)
        best_snap = self._snapshot_positions()
        best_err = float('inf')

        for it in range(max(1, max_rounds)):
            max_abs = 0.0

            for e in self.edges:
                if not e.isVisible(): continue
                u = e.getStartVertex().getIndex()
                v = e.getEndVertex().getIndex()
                pu = self.vertices[u].getPosition()
                pv = self.vertices[v].getPosition()
                dx = pv.x() - pu.x(); dy = pv.y() - pu.y()
                dist = math.hypot(dx, dy)
                if dist < 1e-9: continue

                err = (target_length - dist)  # positive => lengthen
                max_abs = max(max_abs, abs(err))
                if abs(err) <= tol * target_length:
                    continue

                ux, uy = dx / dist, dy / dist

                if (u not in pinned) and (v not in pinned):
                    mu = max(-step, min(step, -0.5 * err))
                    mv = max(-step, min(step,  0.5 * err))
                    self.vertices[u].setPosition(QPointF(pu.x() + ux * mu, pu.y() + uy * mu))
                    self.vertices[v].setPosition(QPointF(pv.x() + ux * mv, pv.y() + uy * mv))
                elif (u in pinned) and (v not in pinned):
                    mv = max(-step, min(step, err))
                    self.vertices[v].setPosition(QPointF(pv.x() + ux * mv, pv.y() + uy * mv))
                elif (v in pinned) and (u not in pinned):
                    mu = max(-step, min(step, -err))
                    self.vertices[u].setPosition(QPointF(pu.x() + ux * mu, pu.y() + uy * mu))
                # both pinned: skip

            self._separate_min_dist_grid(pinned=pinned, dmin=self._min_sep_length(),
                                         iters=separation_iters, clamp_step=step * 1.0)

            if self._has_crossings_quick(cell_size=target_length, max_reports=1):
                self._restore_positions(best_snap)
                step *= 0.6
                if step < 1e-6:
                    break
                continue

            if max_abs < best_err:
                best_err = max_abs
                best_snap = self._snapshot_positions()

            if max_abs <= tol * target_length:
                break

            step = min(step * 1.05, clamp_frac * target_length)

        self._restore_positions(best_snap)

    # --------------------------
    # Local star unit-length projector
    # --------------------------
    def enforce_unit_lengths_local(self, center_idx: int, pinned, target_length: float,
                                   rounds: int = 28, clamp_frac: float = 0.10, tol: float = 9e-4) -> bool:
        neigh = list(self._adjacency.get(center_idx, set()))
        if not neigh or target_length <= 1e-9:
            return True

        step = max(1e-6, clamp_frac * target_length)
        snap = self.vertices[center_idx].getPosition()
        x = snap
        deg = len(neigh)

        for _ in range(max(1, rounds)):
            mvx = mvy = 0.0
            max_abs = 0.0
            for nb in neigh:
                p = self.vertices[nb].getPosition()
                dx = x.x() - p.x()
                dy = x.y() - p.y()
                dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    continue
                err = target_length - dist
                max_abs = max(max_abs, abs(err))
                ux, uy = dx / dist, dy / dist
                mvx += ux * err
                mvy += uy * err

            if deg > 0:
                mvx /= deg
                mvy /= deg

            mvlen = math.hypot(mvx, mvy)
            if mvlen <= tol * target_length * 0.15:
                break

            gamma = 0.85 if deg <= 3 else 0.65
            if mvlen > step:
                s = step / mvlen
                mvx *= s
                mvy *= s

            x = QPointF(x.x() + gamma * mvx, x.y() + gamma * mvy)
            self.vertices[center_idx].setPosition(x)

            if max_abs <= tol * target_length:
                break

        local = set(neigh) | {center_idx}
        self._separate_min_dist_grid(pinned=set(pinned), dmin=self._min_sep_length(),
                                     iters=1, clamp_step=step, only_indices=local)

        if self._has_crossings_quick(max_reports=1):
            self.vertices[center_idx].setPosition(snap)
            return False
        return True

    # --------------------------
    # Tutte relaxation (positive weights, baseline)
    # --------------------------
    def _tutte_relax(self, pinned, rounds=8, jitter_eps=0.03, gauss_seidel=False):
        eps = max(0.0, min(0.2, float(jitter_eps)))

        def hash01(i, j):
            h = (i * 73856093) ^ (j * 19349663)
            h &= 0xffffffff
            return (h % 10007) / 10007.0

        for _ in range(max(1, rounds)):
            if gauss_seidel:
                for i, v in enumerate(self.vertices):
                    if i in pinned or not v.isVisible(): continue
                    neigh = self._adjacency.get(i, set())
                    if not neigh: continue
                    sx = sy = sw = 0.0
                    for nb in neigh:
                        w = 1.0 + eps * (2.0 * hash01(i, nb) - 1.0)
                        p = self.vertices[nb].getPosition()
                        sx += w * p.x(); sy += w * p.y(); sw += w
                    if sw > 1e-12:
                        self.vertices[i].setPosition(QPointF(sx / sw, sy / sw))
            else:
                newpos = {}
                for i, v in enumerate(self.vertices):
                    if i in pinned or not v.isVisible(): continue
                    neigh = self._adjacency.get(i, set())
                    if not neigh: continue
                    sx = sy = sw = 0.0
                    for nb in neigh:
                        w = 1.0 + eps * (2.0 * hash01(i, nb) - 1.0)
                        p = self.vertices[nb].getPosition()
                        sx += w * p.x(); sy += w * p.y(); sw += w
                    if sw > 1e-12:
                        newpos[i] = QPointF(sx / sw, sy / sw)
                for i, p in newpos.items():
                    self.vertices[i].setPosition(p)

    # --------------------------
    # Adaptive positive-weight Tutte (planarity-safe IRLS)
    # --------------------------
    def _tutte_relax_adaptive(self, pinned, target_len, rounds=8, gamma=1.25,
                              wmin=0.35, wmax=3.2, gauss_seidel=True):
        """
        Planarity-safe: positive weights + convex boundary.
        Weights w_ij = clamp( (L_ij / target_len)^gamma, [wmin, wmax] ).
        Long edges (L > target) get larger weights (shrink), short edges get smaller weights (grow).
        """
        if target_len <= 1e-9:
            return

        for _ in range(max(1, rounds)):
            if gauss_seidel:
                # In-place updates (faster convergence)
                for i, v in enumerate(self.vertices):
                    if (i in pinned) or (not v.isVisible()):
                        continue
                    neigh = self._adjacency.get(i, set())
                    if not neigh:
                        continue
                    sx = sy = sw = 0.0
                    pi = v.getPosition()
                    for j in neigh:
                        pj = self.vertices[j].getPosition()
                        dx = pj.x() - pi.x()
                        dy = pj.y() - pi.y()
                        L = math.hypot(dx, dy)
                        r = L / target_len if target_len > 1e-9 else 1.0
                        w = max(wmin, min(wmax, pow(max(1e-9, r), gamma)))
                        sx += w * pj.x()
                        sy += w * pj.y()
                        sw += w
                    if sw > 1e-12:
                        self.vertices[i].setPosition(QPointF(sx / sw, sy / sw))
            else:
                # Jacobi
                newpos = {}
                for i, v in enumerate(self.vertices):
                    if (i in pinned) or (not v.isVisible()):
                        continue
                    neigh = self._adjacency.get(i, set())
                    if not neigh:
                        continue
                    sx = sy = sw = 0.0
                    pi = v.getPosition()
                    for j in neigh:
                        pj = self.vertices[j].getPosition()
                        dx = pj.x() - pi.x()
                        dy = pj.y() - pi.y()
                        L = math.hypot(dx, dy)
                        r = L / target_len if target_len > 1e-9 else 1.0
                        w = max(wmin, min(wmax, pow(max(1e-9, r), gamma)))
                        sx += w * pj.x()
                        sy += w * pj.y()
                        sw += w
                    if sw > 1e-12:
                        newpos[i] = QPointF(sx / sw, sy / sw)
                for i, p in newpos.items():
                    self.vertices[i].setPosition(p)

    # --------------------------
    # Fan spreading
    # --------------------------
    def _spread_fans(self, pinned, angle_min_deg=7.0, iters=3, k_frac=0.06):
        import math
        from qtcore_shim import QPointF as _QPF

        angle_min = math.radians(max(0.1, angle_min_deg))
        target = self._quick_target_length()
        max_step = target * max(0.01, min(0.2, k_frac))

        def v_len_local(a): return math.hypot(a.x(), a.y())

        for _ in range(max(1, iters)):
            forces = {}
            for u in range(len(self.vertices)):
                if not self.vertices[u].isVisible(): continue
                pu = self.vertices[u].getPosition()
                neigh = list(self._adjacency.get(u, set()))
                if len(neigh) < 2: continue

                dirs = []
                for nb in neigh:
                    pv = self.vertices[nb].getPosition()
                    dx = pv.x() - pu.x(); dy = pv.y() - pu.y()
                    L = math.hypot(dx, dy)
                    if L < 1e-6: continue
                    dirs.append((math.atan2(dy, dx), nb, dx / L, dy / L))
                if len(dirs) < 2: continue
                dirs.sort(key=lambda t: t[0])

                m = len(dirs)
                for k in range(m):
                    ang_i, nb_i, dix, diy = dirs[k]
                    ang_j, nb_j, djx, djy = dirs[(k + 1) % m]
                    diff = ang_j - ang_i
                    if diff <= 0: diff += 2 * math.pi
                    if diff >= angle_min: continue

                    cross = dix * djy - diy * djx
                    sign = 1.0 if cross > 0 else -1.0
                    deficit = (angle_min - diff) / angle_min
                    mag = min(max_step, target * 0.06 * deficit)

                    if sign > 0:
                        ti = _QPF(diy * mag, -dix * mag)
                        tj = _QPF(-djy * mag, djx * mag)
                    else:
                        ti = _QPF(-diy * mag, dix * mag)
                        tj = _QPF(djy * mag, -djx * mag)

                    if nb_i not in pinned and self.vertices[nb_i].isVisible():
                        forces[nb_i] = v_add(forces.get(nb_i, _QPF(0, 0)), ti)
                    elif u not in pinned:
                        forces[u] = v_sub(forces.get(u, _QPF(0, 0)), ti)

                    if nb_j not in pinned and self.vertices[nb_j].isVisible():
                        forces[nb_j] = v_add(forces.get(nb_j, _QPF(0, 0)), tj)
                    elif u not in pinned:
                        forces[u] = v_sub(forces.get(u, _QPF(0, 0)), tj)

            for i, f in forces.items():
                fl = v_len_local(f)
                if fl > max_step:
                    f = v_scale(f, max_step / max(1e-9, fl))
                pos = self.vertices[i].getPosition()
                self.vertices[i].setPosition(v_add(pos, f))

    # --------------------------
    # Near-rim spreading
    # --------------------------
    def _spread_near_rim(self, pinned, center, R, angle_min_deg=12.0, iters=2, k_frac=0.07, band=(0.88, 0.98)):
        import math
        from qtcore_shim import QPointF as _QPF

        angle_min = math.radians(max(0.1, angle_min_deg))
        target = self._quick_target_length()
        max_step = target * max(0.01, min(0.25, k_frac))

        def v_len_local(a): return math.hypot(a.x(), a.y())

        for _ in range(max(1, iters)):
            forces = {}
            for u in range(len(self.vertices)):
                if not self.vertices[u].isVisible(): continue
                pu = self.vertices[u].getPosition()
                dx0 = pu.x() - center.x(); dy0 = pu.y() - center.y()
                ru = math.hypot(dx0, dy0)
                if u in pinned or (ru <= band[0] * R or ru >= band[1] * R): continue

                neigh = list(self._adjacency.get(u, set()))
                if len(neigh) < 2: continue

                dirs = []
                for nb in neigh:
                    pv = self.vertices[nb].getPosition()
                    dx = pv.x() - pu.x(); dy = pv.y() - pu.y()
                    L = math.hypot(dx, dy)
                    if L < 1e-6: continue
                    dirs.append((math.atan2(dy, dx), nb, dx / L, dy / L))
                if len(dirs) < 2: continue
                dirs.sort(key=lambda t: t[0])

                tx, ty = dy0 / max(ru, 1e-9), -dx0 / max(ru, 1e-9)

                m = len(dirs)
                for k in range(m):
                    ang_i, nb_i, dix, diy = dirs[k]
                    ang_j, nb_j, djx, djy = dirs[(k + 1) % m]
                    rim_min = math.radians(16.0)
                    nb_i_on_rim = (nb_i in pinned)
                    nb_j_on_rim = (nb_j in pinned)
                    local_min = rim_min if (nb_i_on_rim and nb_j_on_rim) else angle_min

                    diff = ang_j - ang_i
                    if diff <= 0: diff += 2 * math.pi
                    if diff >= local_min: continue

                    cross = dix * djy - diy * djx
                    sgn = 1.0 if cross > 0 else -1.0

                    deficit = (local_min - diff) / local_min
                    mag = min(max_step, target * 0.085 * deficit)

                    tu = _QPF(tx * mag * sgn, ty * mag * sgn)
                    rin = target * 0.05 * deficit
                    ru_safe = max(ru, 1e-9)
                    ru_ix = -dx0 / ru_safe * rin; ru_iy = -dy0 / ru_safe * rin

                    if u not in pinned:
                        forces[u] = v_add(forces.get(u, _QPF(0, 0)), _QPF(tu.x() + ru_ix, tu.y() + ru_iy))
                    else:
                        if nb_i not in pinned and self.vertices[nb_i].isVisible():
                            forces[nb_i] = v_add(forces.get(nb_i, _QPF(0, 0)), _QPF(tx * mag, ty * mag))
                        if nb_j not in pinned and self.vertices[nb_j].isVisible():
                            forces[nb_j] = v_add(forces.get(nb_j, _QPF(0, 0)), _QPF(-tx * mag, -ty * mag))

            for i, f in forces.items():
                fl = v_len_local(f)
                if fl > max_step:
                    f = v_scale(f, max_step / max(1e-9, fl))
                pos = self.vertices[i].getPosition()
                self.vertices[i].setPosition(v_add(pos, f))

    # --------------------------
    # Ultra homogeneity helpers
    # --------------------------
    def _edge_exact_project_axis(self, u: int, v: int, target: float,
                                 clamp_step: float, pinned) -> bool:
        if u in pinned and v in pinned:
            return False
        pu = self.vertices[u].getPosition()
        pv = self.vertices[v].getPosition()
        dx = pv.x() - pu.x(); dy = pv.y() - pu.y()
        dist = math.hypot(dx, dy)  # FIXED (was mistakenly assigning to math.rphypot)
        if dist < 1e-9:
            return False
        ux, uy = dx / dist, dy / dist
        delta = target - dist  # >0 => lengthen, <0 => shorten

        pu_new, pv_new = pu, pv
        if (u not in pinned) and (v not in pinned):
            alpha = 0.5 * delta
            alpha = max(-clamp_step, min(clamp_step, alpha))
            pu_new = QPointF(pu.x() - ux * alpha, pu.y() - uy * alpha)
            pv_new = QPointF(pv.x() + ux * alpha, pv.y() + uy * alpha)
        elif (u in pinned) and (v not in pinned):
            beta = max(-clamp_step, min(clamp_step, delta))
            pv_new = QPointF(pv.x() + ux * beta, pv.y() + uy * beta)
        elif (v in pinned) and (u not in pinned):
            beta = max(-clamp_step, min(clamp_step, -delta))
            pu_new = QPointF(pu.x() - ux * beta, pu.y() - uy * beta)

        snap_u = self.vertices[u].getPosition()
        snap_v = self.vertices[v].getPosition()
        self.vertices[u].setPosition(pu_new)
        self.vertices[v].setPosition(pv_new)

        # Guard: no crossings
        if self._has_crossings_quick(cell_size=target, max_reports=1):
            self.vertices[u].setPosition(snap_u)
            self.vertices[v].setPosition(snap_v)
            return False
        return True

    def polish_homogeneity_ultra(self, pinned, light=False):
        if not self.ultra_homog:
            return
        target = self._quick_target_length()
        if target <= 1e-9 or not self.edges:
            return

        # Gather visible edges with absolute relative error
        edges = []
        for e in self.edges:
            if not e.isVisible():
                continue
            u = e.getStartVertex().getIndex()
            v = e.getEndVertex().getIndex()
            pu = self.vertices[u].getPosition()
            pv = self.vertices[v].getPosition()
            L = math.hypot(pv.x() - pu.x(), pv.y() - pu.y())
            if L > 1e-9:
                err = abs(L / target - 1.0)
                edges.append((err, u, v))

        E = len(edges)
        if E < 2:
            return  # nothing meaningful to adjust

        # For tiny graphs, do a light strict pass and skip the top-K projection
        if E < 6:
            self.enforce_unit_lengths_strict(
                pinned=pinned,
                target_length=target,
                tol=max(self.ultra_tol_small, 8e-4),
                max_rounds=30,
                clamp_frac=min(self.ultra_clamp_frac, 0.10),
                separation_iters=1 if light else 2
            )
            return

        # Sort worst-first
        edges.sort(reverse=True, key=lambda t: t[0])

        # How many edges to project (clamped)
        frac = (self.ultra_edge_proj_frac_small if len(self.vertices) <= self.ultra_small_threshold
                else self.ultra_edge_proj_frac_large)
        if light:
            frac *= 0.6
        K = max(1, int(frac * E))
        K = min(K, E)  # CLAMP to avoid IndexError

        clamp_step = self.ultra_clamp_frac * target
        changed = False
        for k in range(K):
            _, u, v = edges[k]
            if self._edge_exact_project_axis(u, v, target, clamp_step, pinned):
                changed = True

        if changed:
            # Light separation to keep nodes from colliding
            self._separate_min_dist_grid(pinned=pinned, dmin=self._min_sep_length(),
                                        iters=1 if light else 2, clamp_step=target * 0.08)

        # Finish with a final strict pass (tighter) to polish
        tol = self.ultra_tol_small if len(self.vertices) <= self.ultra_small_threshold else self.ultra_tol_large
        rounds = self.ultra_rounds_small if len(self.vertices) <= self.ultra_small_threshold else self.ultra_rounds_large
        clamp_frac = min(self.ultra_clamp_frac, 0.10 if light else self.ultra_clamp_frac)
        self.enforce_unit_lengths_strict(
            pinned=pinned,
            target_length=target,
            tol=tol,
            max_rounds=rounds,
            clamp_frac=clamp_frac,
            separation_iters=1 if light else 2
        )

    # --------------------------
    # Homogeneity quick metric (p95)
    # --------------------------
    def _homog_p95_ratio(self) -> float:
        T = self._quick_target_length()
        if T <= 1e-9 or not self.edges:
            return 1.0
        ratios = []
        for e in self.edges:
            if not e.isVisible():
                continue
            u = e.getStartVertex().getIndex()
            v = e.getEndVertex().getIndex()
            p1 = self.vertices[u].getPosition()
            p2 = self.vertices[v].getPosition()
            L = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            ratios.append(L / T)
        if not ratios:
            return 1.0
        ratios.sort()
        i = min(len(ratios) - 1, int(0.95 * len(ratios)))
        return ratios[i]

    def _auto_polish_after_add(self, pinned):
        # Up to N focused polish passes until p95 is tight enough
        for _ in range(max(0, int(self.auto_polish_max_passes))):
            if self._homog_p95_ratio() <= float(self.auto_polish_p95):
                break
            T = self._quick_target_length()
            # Strong planarity-safe push toward unit lengths
            self._tutte_relax_adaptive(
                pinned=pinned,
                target_len=T,
                rounds=7,
                gamma=1.35,
                wmin=0.30, wmax=4.0,
                gauss_seidel=True
            )
            # Clamp very long edges and gently expand short ones
            # Be more aggressive on very long edges, slightly widen tolerance band
            self._long_edge_reduce(pinned, T, hi1=1.12, hi2=1.80,
                                   gain1=0.65, gain2=0.90,
                                   max_step=0.22 * T, rounds=1)
            self._edge_length_homogenize(pinned, T, lo=0.96, hi=1.06,
                                         gain=0.62, max_step=0.14 * T, rounds=1)
            # Angle/fan spread and vertex–edge clearance to remove near-parallel stacks
            self._spread_fans(pinned=pinned, angle_min_deg=10.0, iters=1, k_frac=0.06)
            self._ve_clearance_grid(pinned=pinned, clear_dist=0.22 * T, iters=1, clamp_step=0.10 * T)
            # If congestion persists anywhere, expand globally a touch (planarity safe)
            self._global_expand_if_congested(inflate_pad=1.03)
            # Tight strict pass with guard
            self.enforce_unit_lengths_strict(
                pinned=pinned,
                target_length=T,
                tol=4e-4,
                max_rounds=90,
                clamp_frac=0.10,
                separation_iters=2
            )
            # Keep a little spacing and recheck congestion
            self._separate_min_dist_grid(pinned=pinned, dmin=self._min_sep_length(), iters=1)
            self._global_expand_if_congested(inflate_pad=1.02)

    # --------------------------
    # Redraw (global)
    # --------------------------
# graph.py

    def redraw_planar(self, iterations=None, radius=None, light=False):
        """
        Planarity-safe redraw: do a Tutte re-embed directly (guaranteed planar),
        then an optional tiny positive-weight polish.
        """
        per = self.periphery.getIndices()
        if len(per) < 3:
            return

        # Choose SciPy for larger interior solves
        n_interior = max(0, len(self.vertices) - len(per))
        use_scipy = (n_interior > 150)

        # Direct Tutte embed
        self.reembed_tutte(radius=radius, use_scipy=use_scipy, final_polish=True if not light else False)
        # After a global re-embed, ensure no congestion remains by scaling out slightly if needed
        self._global_expand_if_congested(inflate_pad=1.02)
    # --------------------------
    # Normalize & recenter
    # --------------------------
    def _normalize_and_recenter(self, max_radius=None):
        per = self.periphery.getIndices()
        if not per: return
        center = self._periphery_center()

        tx = -center.x(); ty = -center.y()
        if abs(tx) > 1e-6 or abs(ty) > 1e-6:
            for v in self.vertices:
                p = v.getPosition()
                self.vertices[v.getIndex()].setPosition(QPointF(p.x() + tx, p.y() + ty))

        if max_radius is None:
            return

        max_r = 0.0
        for v in self.vertices:
            p = v.getPosition()
            r = math.hypot(p.x(), p.y())
            if r > max_r: max_r = r

        if max_r > max_radius:
            s = max_radius / max(1e-9, max_r)
            for v in self.vertices:
                p = v.getPosition()
                self.vertices[v.getIndex()].setPosition(QPointF(p.x() * s, p.y() * s))
            if self._target_len:
                self._target_len *= s

    # --------------------------
    # Validation and stats
    # --------------------------
    def validate_coloring(self):
        bad = []
        for e in self.edges:
            u = e.getStartVertex().getIndex()
            v = e.getEndVertex().getIndex()
            if self.vertices[u].getColorIndex() == self.vertices[v].getColorIndex():
                bad.append((u, v))
        return bad

    def get_stats(self):
        seed = sum(1 for v in self.vertices if v and v.getOrigin() == "seed")
        random_cnt = sum(1 for v in self.vertices if v and v.getOrigin() == "random")
        manual = sum(1 for v in self.vertices if v and v.getOrigin() == "manual")
        return {
            "total_vertices": len(self.vertices),
            "seed": seed,
            "random": random_cnt,
            "manual": manual,
            "edges": len(self.edges),
            "periphery_size": len(self.periphery.getIndices()),
        }

    def homogeneity_report(self, visible_only=True):
        lens = []
        for e in self.edges:
            if visible_only and not e.isVisible():
                continue
            p1 = e.getStartVertex().getPosition()
            p2 = e.getEndVertex().getPosition()
            lens.append(v_len(v_sub(p2, p1)))
        if not lens:
            return {}

        T = self._quick_target_length()
        if T <= 1e-9:
            return {"edges": len(lens), "target_length": T}

        ratios = [L / T for L in lens]
        ratios_sorted = sorted(ratios)
        p95 = ratios_sorted[int(max(0, min(len(ratios_sorted) - 1, 0.95 * len(ratios_sorted))))]
        from statistics import mean, pstdev
        return {
            "edges": len(lens),
            "target_length": T,
            "mean_ratio": mean(ratios),
            "std_ratio": pstdev(ratios) if len(ratios) > 1 else 0.0,
            "p95_ratio": p95,
            "min_ratio": ratios_sorted[0],
            "max_ratio": ratios_sorted[-1],
        }

    # --------------------------
    # Persistence
    # --------------------------

    def save_to_json(self, path) -> bool:
        """
        Save the graph to JSON in a fully serializable format.
        - Vertices: index, x, y, colorIndex, origin, visible
        - Edges: list of [u, v] index pairs
        - Periphery: CW list of vertex indices
        - Label mode
        Returns True on success, False on failure.
        """
        try:
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
                    for v in self.vertices
                    if v is not None
                ],
                "edges": [
                    [e.getStartVertex().getIndex(), e.getEndVertex().getIndex()]
                    for e in self.edges
                ],
                "periphery": list(self.periphery.getIndices()),
                "labelMode": int(self.labelMode),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as ex:
            print(f"save_to_json failed: {ex}")
            return False


    def load_from_json(self, path) -> bool:
        """
        Load graph from JSON saved by save_to_json().
        - Rebuilds vertices, edges, adjacency, and periphery.
        - Preserves saved positions and visibility.
        - Returns True on success, False on failure.
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Reset all structures
            self.clear()

            # 1) Vertices
            verts = data.get("vertices", [])
            if not isinstance(verts, list) or not verts:
                raise ValueError("JSON missing 'vertices' list.")

            max_idx = max(int(v["index"]) for v in verts)
            self.vertices = [None] * (max_idx + 1)

            for rec in verts:
                idx = int(rec["index"])
                x = float(rec["x"]); y = float(rec["y"])
                color_idx = int(rec.get("colorIndex", 1))
                origin = rec.get("origin", "manual")
                visible = bool(rec.get("visible", True))

                v = Vertex(idx, QPointF(x, y), colorIndex=color_idx, origin=origin)
                v.setVisible(visible)
                self.vertices[idx] = v
                self._adjacency[idx] = set()

            # 2) Edges (as Edge objects, rebuild adjacency)
            edges_list = data.get("edges", [])
            if not isinstance(edges_list, list):
                edges_list = []

            for uv in edges_list:
                if not isinstance(uv, (list, tuple)) or len(uv) != 2:
                    continue
                u, v = int(uv[0]), int(uv[1])
                if 0 <= u < len(self.vertices) and 0 <= v < len(self.vertices):
                    if self.vertices[u] is not None and self.vertices[v] is not None and u != v:
                        e = Edge(self.vertices[u], self.vertices[v])
                        # Edge visibility = both endpoints visible
                        e.setVisible(self.vertices[u].isVisible() and self.vertices[v].isVisible())
                        self._add_edge_internal(e)

            # 3) Periphery (CW)
            peri = [int(i) for i in data.get("periphery", []) if 0 <= int(i) < len(self.vertices) and self.vertices[int(i)] is not None]
            if len(peri) >= 3 and len(set(peri)) == len(peri):
                self.periphery.initialize(peri)
            else:
                # Fallback: try first 3 existing vertices (still CW order unknown)
                base = [i for i, v in enumerate(self.vertices) if v is not None][:3]
                if len(base) == 3:
                    self.periphery.initialize(base)
                else:
                    raise ValueError("Cannot rebuild periphery: not enough vertices.")

            # 4) Label mode (optional)
            self.labelMode = int(data.get("labelMode", 2))

            # 5) Update derived state
            # Do NOT re-embed — keep saved positions; just recompute target length and edge visibilities
            for e in self.edges:
                u = e.getStartVertex().getIndex()
                v = e.getEndVertex().getIndex()
                e.setVisible(self.vertices[u].isVisible() and self.vertices[v].isVisible())

            self._target_len = self._compute_target_edge_length()

            # Light sanity check: periphery index map valid
            if not self.periphery.validate():
                print("Warning: periphery failed validation after load.")
            return True

        except Exception as ex:
            print(f"load_from_json failed: {ex}")
            return False


    # --------------------------
    # Getters (used by UI)
    # --------------------------
    def getVertices(self):
        return self.vertices

    def getEdges(self):
        return self.edges

    def getPeriphery(self):
        return self.periphery.getIndices()

    # --- Label mode helpers ---
    def get_label_mode(self) -> int:
        return self.labelMode

    def cycle_label_mode(self) -> int:
        self.labelMode = (self.labelMode + 1) % 3
        return self.labelMode

    # --- Animation info for last addition ---
    def get_last_add_info(self):
        return self._last_add_info

    # --------------------------
    # World/UI helpers and flags
    # --------------------------

    def set_max_working_radius(self, limit: Optional[float]):
        self._MAX_WORKING_RADIUS = float(limit) if (limit is not None) else None

    def get_max_working_radius(self) -> Optional[float]:
        return self._MAX_WORKING_RADIUS

    def get_bounding_box(self, visible_only=True):
        xs = []; ys = []
        for v in self.vertices:
            if visible_only and not v.isVisible(): continue
            p = v.getPosition()
            xs.append(p.x()); ys.append(p.y())
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def get_center(self) -> QPointF:
        return self._periphery_center()

    def set_auto_expand(self, enabled: bool):
        self.auto_expand = bool(enabled)

    def get_auto_expand(self) -> bool:
        return self.auto_expand

    def set_auto_expand_mode(self, mode: str):
        mode_l = (mode or "").strip().lower()
        if mode_l in ("fit", "infinite"):
            self.auto_expand_mode = mode_l

    def get_auto_expand_mode(self) -> str:
        return self.auto_expand_mode

    def set_view_padding(self, px: float):
        self.view_padding = float(px)

    def get_view_padding(self) -> float:
        return self.view_padding

    def set_target_edge_px(self, px: float):
        self.target_edge_px = float(px)

    def get_target_edge_px(self) -> float:
        return self.target_edge_px

    # --------------------------
    # Coloring helpers (restored)
    # --------------------------
    def _greedy_color(self, vid: int):
        # Choose the smallest color in {1..4} not used by neighbors
        used = {self.vertices[n].getColorIndex() for n in self._adjacency.get(vid, set())}
        c = 1
        while c in used and c < 4:
            c += 1
        self.vertices[vid].setColorIndex(c)

    def setVertexColor(self, vertex_index, color_id):
        try:
            cid = int(color_id)
        except Exception:
            return False
        if not (1 <= cid <= 4):
            return False
        if 0 <= vertex_index < len(self.vertices):
            self.vertices[vertex_index].setColorIndex(cid)
            return True
        return False
    

    def validate_invariants(self, verbose: bool = False) -> bool:
        ok = True
        per = list(self.periphery.getIndices())
        h = len(per)
        V = len(self.vertices)
        E = len(self.edges)

        # Periphery mapping and uniqueness
        if not self.periphery.validate():
            ok = False
            if verbose:
                print("Periphery map inconsistent or contains duplicates.")

        # Each consecutive periphery pair must be adjacent
        for i in range(h):
            u = per[i]
            v = per[(i + 1) % h]
            if v not in self._adjacency.get(u, set()):
                ok = False
                if verbose:
                    print(f"Missing periphery edge ({u}, {v}).")

        # Adjacency symmetry and no self-loops
        for u, nbrs in self._adjacency.items():
            if u < 0 or u >= V or self.vertices[u] is None:
                ok = False
                if verbose:
                    print(f"Bad vertex in adjacency: {u}")
                continue
            if u in nbrs:
                ok = False
                if verbose:
                    print(f"Self-loop at {u}")
            for v in nbrs:
                if v < 0 or v >= V or self.vertices[v] is None:
                    ok = False
                    if verbose:
                        print(f"Invalid neighbor {v} for {u}")
                    continue
                if u not in self._adjacency.get(v, set()):
                    ok = False
                    if verbose:
                        print(f"Asymmetry: {u} has {v}, but {v} missing {u}")

        # Triangulated planar graph with outer face size h should satisfy E = 3V - 3 - h
        if V >= 3 and h >= 3:
            expected_E = 3 * V - 3 - h
            if E != expected_E:
                ok = False
                if verbose:
                    print(f"E={E}, expected={expected_E} (V={V}, h={h})")

        # Removed incorrect call: self.periphery.isContiguous() with no arguments
        return ok
    
    def get_vertex_degree(self, vid: int) -> int:
        """
        Return the degree (number of incident edges) of a vertex by index.
        Uses adjacency for O(1) lookup and validates bounds.
        """
        if not (0 <= vid < len(self.vertices)) or self.vertices[vid] is None:
            return 0
        return len(self._adjacency.get(vid, set()))

    def declutter_for_view(self, intensity: float = 1.0) -> bool:
        """
        Gentle, planarity-safe spacing pass.
        - Keeps periphery pinned (convex), uses positive-weight Tutte relaxation,
        then applies small vertex-vertex and vertex-edge clearance.
        - Crossing guard: if crossings appear, revert and retry with smaller steps.
        Returns True if layout changed, False otherwise.
        """
        per = list(self.periphery.getIndices())
        if len(per) < 3:
            return False

        pinned = set(per)
        T = self._quick_target_length()
        if T <= 1e-9:
            T = 60.0

        snap = self._snapshot_positions()
        changed = False
        try:
            # 1) Positive-weight smoothing (shrinks long edges, grows short ones)
            rounds = max(1, int(3 + 2 * float(intensity)))
            self._tutte_relax_adaptive(
                pinned=pinned, target_len=T, rounds=rounds,
                gamma=1.28, wmin=0.35, wmax=3.0, gauss_seidel=True
            )

            # 2) Vertex-vertex spacing
            dmin = max(self._min_sep_length(), 0.22 * T) * (1.0 + 0.35 * float(intensity))
            self._separate_min_dist_grid(
                pinned=pinned, dmin=dmin, iters=1 + int(max(0, int(float(intensity)))),
                clamp_step=0.10 * T
            )

            # 3) Vertex-edge clearance
            self._ve_clearance_grid(
                pinned=pinned, clear_dist=0.26 * T, iters=2, clamp_step=0.10 * T
            )

            # Crossing guard
            if self._has_crossings_quick(cell_size=T, max_reports=1):
                # revert and retry smaller
                self._restore_positions(snap)
                self._separate_min_dist_grid(pinned=pinned, dmin=0.85 * dmin, iters=1, clamp_step=0.08 * T)
                self._ve_clearance_grid(pinned=pinned, clear_dist=0.16 * T, iters=1, clamp_step=0.08 * T)
                if self._has_crossings_quick(cell_size=T, max_reports=1):
                    self._restore_positions(snap)
                    return False

            changed = True
            return True
        finally:
            if changed:
                
                if self.on_layout_changed:
                    self.on_layout_changed(self.get_bounding_box(), self._periphery_center(), self.target_edge_px)
