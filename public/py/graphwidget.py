# graphwidget.py

from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene,
    QGraphicsSimpleTextItem, QMessageBox, QFileDialog, QMenu,QGraphicsItem,
)

from PyQt5.QtCore import Qt, QPointF, QTimeLine, QRectF
from PyQt5.QtGui import QPen, QColor, QPainter, QImage, QPainterPath, QTransform,QFont
from PyQt5.QtWidgets import QGraphicsScene as QGS
from typing import Optional
from graph import Graph
from utils_geom import v_add, v_sub, v_scale, v_len, v_lerp, v_mid
import math
from PyQt5.QtGui import QPen, QColor, QPainter, QImage, QPainterPath, QTransform, QFont, QFontMetricsF

# Zoom behavior constants (stabilized)
ZOOM_FACTOR = 1.15
ZOOM_MAX = 5000.0
MIN_ABS_SCALE = 1e-3
MIN_REL_TO_FIT = 0.25

# Animation tuning (60 FPS + slower center)
ANIM_FPS = 60
ANIM_DT_MS = 16                  # ~60 FPS (1000/60 ≈ 16.67; Qt uses ints, so 16 is closest)
ADD_ANIM_MS = 420                # vertex position animation duration
CENTER_ANIM_MS = 600             # center-graph camera animation duration (slower, smoother)
REFRAME_ANIM_MS = 480            # if you ever call smart reframe

# Strict planar render: draw all edges as straight lines (no curves)
STRICT_PLANAR_RENDER = False

# Fixed curve angles (degrees) - used only if STRICT_PLANAR_RENDER = False
PERIPHERY_CURVE_DEG = 30.0
INTERIOR_CURVE_DEG = 15.0

NEW_GRAPH_ANIM_MS = 600

class GraphWidget(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph = Graph()
        scene = QGraphicsScene(self)
        scene.setItemIndexMethod(QGS.NoIndex)
        self.setScene(scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheBackground)

        self.currentZoom = 1.0
        self._base_fit_scale = None
        self.selectingPeriphery = False
        self.peripheryStartIndex = -1
        self.panning = False
        self.lastPanPoint = None

        self._cameraTimeline = None
        self._highlightTimeline = None
        self._highlightedVertex = -1
        self._highlightProgress = 0.0
        self._animTimeline = None  # vertex position animation

        self.SAFE_MODE = True
        self._in_update_scene = False

        self.palette_outline = [
            QColor("#e74c3c"),
            QColor("#3498db"),
            QColor("#2ecc71"),
            QColor("#f1c40f"),
        ]
        self.palette_fill = [
            QColor("#ff6b6b"),
            QColor("#48dbfb"),
            QColor("#1dd1a1"),
            QColor("#feca57"),
        ]

        # Clean background (no grid)
        self.gridEnabled = False
        self.gridBackground = QColor(255, 255, 255)

        # Aggregation thresholds (10k+ friendly)
        self.AGGREGATE_EDGES_FROM = 2500
        self.AGGREGATE_VERTICES_FROM = 2500

        # Hook: ignore auto camera moves during animations
        self.graph.on_layout_changed = self._onLayoutChanged
        self.curvedPeripheryEnabled = True
        self._pending_random_adds = 0
        self._waiting_for_camera_finish = False
        
        #theme
        self.themeMode = 'light'
        self.edgeColor = QColor(60, 60, 60)
        self.labelIndexOnlyColor = QColor(0, 0, 0)
        self.setTheme('light')  # default; MainWindow.applyTheme will override
                # Zoom-aware node drawing
        self.NODE_SHRINK_GAMMA = 1.20   # >1 shrinks nodes as you zoom in (tweak: 1.1..1.5)
        self.NODE_MIN_PX = 6.0          # clamp min node diameter in pixels
        self.NODE_MAX_PX = 44.0         # clamp max node diameter in pixels
        self.LABEL_NODE_FRACTION = 0.60 # index text is 30% of node diameter (on-screen)

        # Auto declutter when zoomed-in (planarity-safe, small local spread)
        self.AUTO_DECLUTTER_ON_ZOOM = True
        self.DECLUTTER_TRIGGER_SCALE = 1.6   # run declutter when zoom >= this
        self.DECLUTTER_INTENSITY = 1.0       # 0.7..1.5 are good values
        self.finalizeMode = False
        self.selectionCwMode = 'visual' # or we can select visual 
        
    # --------------------------
    # Animation and theme helpers Selection CW mode helpers (Program vs Visual)
    # --------------------------

    def setSelectionCwMode(self, mode: str):
        m = (mode or '').strip().lower()
        if m in ('program', 'visual'):
            self.selectionCwMode = m
            try:
                self.parent().statusBar().showMessage(
                    f"Selection CW mode: {'Visual (screen)' if m == 'visual' else 'Program (stored)'}", 2500
                )
            except Exception:
                pass

    def toggleSelectionCwMode(self):
        self.setSelectionCwMode('visual' if self.selectionCwMode == 'program' else 'program')

    def _angle_up(self, p: QPointF, c: QPointF):
        """
        Angle from center->point in a math y-up system (Qt is y-down).
        """
        import math
        return math.atan2(-(p.y() - c.y()), (p.x() - c.x()))

    def _visual_cw_is_stored_next(self, vp_idx: int) -> bool:
        """
        True if the stored 'next' neighbor of vp_idx is visually clockwise on screen.
        Uses angles around periphery center in y-up coordinates (robust, layout independent).
        """
        prev_u, next_u = self.graph.periphery.neighborsOnPeriphery(vp_idx)
        if prev_u is None or next_u is None:
            return True  # default safe choice

        c = self.graph.get_center()
        pv = self.graph.vertices[vp_idx].getPosition()
        pp = self.graph.vertices[prev_u].getPosition()
        pn = self.graph.vertices[next_u].getPosition()

        import math
        tv = self._angle_up(pv, c)
        tp = self._angle_up(pp, c)
        tn = self._angle_up(pn, c)

        # Normalize angle delta into [-pi, pi]
        def dtheta(a, b):
            d = a - b
            while d <= -math.pi:
                d += 2.0 * math.pi
            while d > math.pi:
                d -= 2.0 * math.pi
            return d

        # In y-up math, clockwise = negative delta when moving from tv to neighbor angle
        is_next_cw = dtheta(tn, tv) < 0.0
        return is_next_cw

    def _cw_neighbors_hint(self, vp_idx: int):
        """
        Return (cw_neighbor, ccw_neighbor) as per the active mode.
        - program: cw = stored next, ccw = stored prev
        - visual:  cw = whichever (prev/next) is clockwise on screen by angle test
        """
        prev_u, next_u = self.graph.periphery.neighborsOnPeriphery(vp_idx)
        if prev_u is None or next_u is None:
            return None, None

        if self.selectionCwMode == 'program':
            return next_u, prev_u

        # visual mode: pick the one that is visually clockwise on screen
        return (next_u, prev_u) if self._visual_cw_is_stored_next(vp_idx) else (prev_u, next_u)

    def _map_selection_to_program_cw(self, a: int, b: int):
        """
        Map user's (a -> b) to the program's CW call.
        - program mode: (a, b) unchanged
        - visual mode: if stored-next is not visually CW from 'a', swap so visual CW becomes stored CW
        """
        if self.selectionCwMode != 'visual':
            return a, b

        # If stored-next is visually CW, keep (a,b); else swap
        swap = not self._visual_cw_is_stored_next(a)
        return (b, a) if swap else (a, b)
    def setFinalizeMode(self, enabled: bool):
        """Receive finalize-mode state from MainWindow and enforce locally."""
        self.finalizeMode = bool(enabled)
        # If user was selecting an arc, cancel selection when finalize is turned ON
        if self.finalizeMode and self.selectingPeriphery:
            self.cancelSelection()
        # Just a small feedback
        try:
            state = "ON" if self.finalizeMode else "OFF"
            self.parent().statusBar().showMessage(f"Finalize mode: {state}", 2000)
        except Exception:
            pass
    
    def tutteReembedNow(self):
        """Manually trigger a Tutte re-embed and animate to the new layout."""
        per = self.graph.periphery.getIndices()
        if len(per) < 3:
            try:
                self.parent().statusBar().showMessage("Need at least a triangle to re-embed.", 3000)
            except Exception:
                pass
            return
        pre = self._snapshot_positions()
        try:
            n_interior = max(0, len(self.graph.vertices) - len(per))
            self.graph.reembed_tutte(use_scipy=(n_interior > 150), final_polish=True)
        except Exception:
            self.graph.reembed_tutte(use_scipy=False, final_polish=True)
        post = self._snapshot_positions()
        # Animate transition and recentre
        self._animate_transition(pre, post, duration_ms=ADD_ANIM_MS)
        try:
            self.parent().statusBar().showMessage("Tutte re-embed complete.", 2500)
        except Exception:
            pass

    def setAutoTutteEmbed(self, enabled: bool):
        """Enable/disable auto Tutte re-embed on each add."""
        self.graph.set_auto_tutte_reembed(bool(enabled))
        try:
            state = "ON" if enabled else "OFF"
            self.parent().statusBar().showMessage(f"Auto Tutte Re-Embed: {state}", 2500)
        except Exception:
            pass
    
    def _node_pixel_diameter(self, base_scene_d: float, scale_now: float) -> float:
        """
        Compute desired on-screen node diameter (in pixels).
        Shrinks as zoom increases using exponent gamma (>1).
        """
        gamma = getattr(self, "NODE_SHRINK_GAMMA", 1.20)
        min_px = getattr(self, "NODE_MIN_PX", 6.0)
        max_px = getattr(self, "NODE_MAX_PX", 44.0)
        # Start from how big the node would appear at this zoom, then shrink by scale^gamma
        px = (base_scene_d * scale_now) / max(pow(scale_now, gamma), 1.0)
        return max(min_px, min(max_px, px))

    def _maybeDeclutterAfterZoom(self):
        """
        When zoomed in enough, gently increase spacing with a planarity-safe pass.
        """
        if not getattr(self, "AUTO_DECLUTTER_ON_ZOOM", False):
            return
        s = self._scaleNow()
        if s < getattr(self, "DECLUTTER_TRIGGER_SCALE", 1.6):
            return
        try:
            changed = self.graph.declutter_for_view(intensity=getattr(self, "DECLUTTER_INTENSITY", 1.0))
            if changed:
                self.updateGraphScene()
        except Exception:
            # Fail-safe: never break UI on declutter issues
            pass
    def togglePeripheryCurves(self):
        """
        Toggle between curved and straight rendering for outer (periphery) edges.
        """
        self.curvedPeripheryEnabled = not self.curvedPeripheryEnabled
        try:
            mode = "curved" if self.curvedPeripheryEnabled else "straight"
            self.parent().statusBar().showMessage(f"Outer edges: {mode}", 2000)
        except Exception:
            pass
        self.updateGraphScene()
    def setTheme(self, mode: str):
        """
        Apply theme to the canvas (background, edge color, label tint for index-only mode).
        mode: 'light' or 'dark'
        """
        m = (mode or '').strip().lower()
        if m == 'dark':
            self.themeMode = 'dark'
            self.gridBackground = QColor(18, 18, 18)     # canvas background
            self.edgeColor = QColor(185, 185, 185)       # lighter edges on dark bg
            self.labelIndexOnlyColor = QColor(0, 0, 0)   # index-only labels on white nodes
        else:
            self.themeMode = 'light'
            self.gridBackground = QColor(255, 255, 255)
            self.edgeColor = QColor(60, 60, 60)
            self.labelIndexOnlyColor = QColor(0, 0, 0)

        self.updateGraphScene()
    def _makeTimeline(self, duration_ms: int) -> QTimeLine:
        tl = QTimeLine(int(duration_ms), self)
        tl.setUpdateInterval(ANIM_DT_MS)  # ~60 FPS
        tl.setCurveShape(QTimeLine.EaseInOutCurve)
        return tl

    def _snapshot_positions(self):
        snap = {}
        for v in self.graph.getVertices():
            if not v:
                continue
            p = v.getPosition()
            snap[v.getIndex()] = QPointF(p.x(), p.y())
        return snap

    def _apply_positions_snapshot(self, snap):
        for idx, p in snap.items():
            if 0 <= idx < len(self.graph.vertices) and self.graph.vertices[idx]:
                self.graph.vertices[idx].setPosition(p)

    def _apply_interpolated_positions(self, pre, post, t):
        keys = set(pre.keys()) | set(post.keys())
        for idx in keys:
            a = pre.get(idx, post.get(idx))
            b = post.get(idx, pre.get(idx))
            if a is None or b is None:
                continue
            pos = v_lerp(a, b, t)
            self.graph.vertices[idx].setPosition(pos)
        self.updateGraphScene()

    def _animate_transition(self, pre, post, duration_ms=ADD_ANIM_MS):
        # Stop camera anim if running to avoid blending
        if self._cameraTimeline and self._cameraTimeline.state() == QTimeLine.Running:
            self._cameraTimeline.stop()
        if self._highlightTimeline and self._highlightTimeline.state() == QTimeLine.Running:
            self._highlightTimeline.stop()

        # Start from 'pre'
        self._apply_positions_snapshot(pre)
        self.updateGraphScene()

        tl = self._makeTimeline(duration_ms)
        tl.valueChanged.connect(lambda t: self._apply_interpolated_positions(pre, post, t))

        def finish():
            # Snap to final layout
            self._apply_positions_snapshot(post)
            self.updateGraphScene()
            # Smoothly center to the exact view that centerGraph() would produce
            self.animateCenterGraph(duration_ms=CENTER_ANIM_MS)

        tl.finished.connect(finish)
        tl.start()
        self._animTimeline = tl

    def animateCenterGraph(self, duration_ms=CENTER_ANIM_MS):
        # Equivalent final state to centerGraph(), but animated (no jerk)
        if not self.scene().items():
            return
        rect = self.scene().itemsBoundingRect()
        adjust = max(50, len(self.graph.getVertices()) / 100)
        target_rect = rect.adjusted(-adjust, -adjust, adjust, adjust)
        self._animateToSceneRect(target_rect, duration_ms)

    def _animateToSceneRect(self, target_rect: QRectF, duration_ms=REFRAME_ANIM_MS):
        if target_rect.isEmpty():
            return
        start_transform = self.transform()
        self.fitInView(target_rect, Qt.KeepAspectRatio)
        end_transform = self.transform()
        self.setTransform(start_transform)

        if self._cameraTimeline and self._cameraTimeline.state() == QTimeLine.Running:
            self._cameraTimeline.stop()

        timeline = self._makeTimeline(duration_ms)
        timeline.valueChanged.connect(
            lambda t: self.setTransform(
                self._interpolateTransform(start_transform, end_transform, t)
            )
        )
        def _on_cam_finish():
            self._base_fit_scale = self.transform().m11()
            self.updateGraphScene()
            self._maybeDeclutterAfterZoom()
        timeline.finished.connect(_on_cam_finish)
        timeline.start()
        self._cameraTimeline = timeline

    def _interpolateTransform(self, a: QTransform, b: QTransform, t: float) -> QTransform:
        t = max(0.0, min(1.0, t))
        m11 = a.m11() + (b.m11() - a.m11()) * t
        m22 = a.m22() + (b.m22() - a.m22()) * t
        m12 = a.m12() + (b.m12() - a.m12()) * t
        m21 = a.m21() + (b.m21() - a.m21()) * t
        dx = a.m31() + (b.m31() - a.m31()) * t
        dy = a.m32() + (b.m32() - a.m32()) * t
        return QTransform(m11, m12, m21, m22, dx, dy)

    # Optional: runtime control from your UI
    def setAnimationFps(self, fps: int):
        global ANIM_FPS, ANIM_DT_MS
        ANIM_FPS = int(max(1, fps))
        ANIM_DT_MS = max(1, int(round(1000.0 / ANIM_FPS)))

    def setAnimationDurations(self, add_ms: Optional[int] = None,
                              center_ms: Optional[int] = None,
                              reframe_ms: Optional[int] = None):
        global ADD_ANIM_MS, CENTER_ANIM_MS, REFRAME_ANIM_MS
        if add_ms is not None: ADD_ANIM_MS = int(max(1, add_ms))
        if center_ms is not None: CENTER_ANIM_MS = int(max(1, center_ms))
        if reframe_ms is not None: REFRAME_ANIM_MS = int(max(1, reframe_ms))

    # --------------------------
    # Auto-layout hook (disable during animations)
    # --------------------------
    def _onLayoutChanged(self, bbox_tuple, center_point: QPointF, target_edge_px: float):
        # Avoid camera changes mid-vertex animation
        if self._animTimeline and self._animTimeline.state() == QTimeLine.Running:
            return
        # We intentionally do nothing here; camera is centered smoothly after each add.

    # --------------------------
    # Graph lifecycle
    # --------------------------
    def setGridEnabled(self, flag: bool):
        self.gridEnabled = bool(flag)
        self.viewport().update()

    def confirmStartGraph(self):
        # Ask for confirmation only if we already have a graph with > 3 vertices
        if len(self.graph.getVertices()) > 3:
            reply = QMessageBox.question(
                self, "Confirm Reset",
                "This will reset the current graph. Are you sure?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Prompt for number of seed vertices (3..10 per client spec)
        from PyQt5.QtWidgets import QInputDialog
        n, ok = QInputDialog.getInt(
            self, "New Graph",
            "Number of seed vertices (3..10):",
            value=3, min=3, max=10
        )
        if not ok:
            return
        self.startBasicGraph(n)

    def startBasicGraph(self, n: int = 3):
        # Reset view
        self.resetTransform()
        self.currentZoom = 1.0

        # Build the graph (final positions are now set on a circle)
        self.graph.startBasicGraph(n)

        # Update GoTo control
        try:
            self.parent().spinGoTo.setMaximum(100000)
            self.parent().spinGoTo.setValue(len(self.graph.getVertices()))
        except Exception:
            pass

        # Prepare animation: collapse all vertices near the center, then expand to final
        post = self._snapshot_positions()              # final layout
        center = self.graph.get_center()               # periphery center
        spawn_ratio = 0.05                              # 5% of final radial distance (tweakable)
        pre = {}
        for idx, p in post.items():
            dx = p.x() - center.x()
            dy = p.y() - center.y()
            pre[idx] = QPointF(center.x() + spawn_ratio * dx,
                            center.y() + spawn_ratio * dy)

        # Animate from pre -> post, then smoothly center the camera
        self._animate_transition(pre, post, duration_ms=NEW_GRAPH_ANIM_MS)

    def addRandomVertex(self):
        """
        Request to add a random vertex. If an animation is in progress,
        the request is queued and will run as soon as the graph settles.
        """
        # If an animation is running (vertex or camera), queue the request
        if self._isBusyAnimating():
            self._pending_random_adds += 1
            try:
                self.parent().statusBar().showMessage(
                    f"Queued random add (x{self._pending_random_adds}). Will run after animation.",
                    2000
                )
            except Exception:
                pass
            return

        # No animation running — start the queued sequence with 1 request
        self._pending_random_adds += 1
        self._attempt_process_random_queue()

    def addVertexBySelection(self):
        """
        Enter interactive mode: user clicks Vp then Vq on the periphery to define the CW arc.
        Insertion is executed in mousePressEvent once both endpoints are picked.
        """
        self.selectingPeriphery = True
        self.peripheryStartIndex = -1
        self.setCursor(Qt.PointingHandCursor)
        try:
            self.parent().statusBar().showMessage(
                "Select Vp on the periphery, then Vq (clockwise arc). Click outside to cancel.",
                6000
            )
        except Exception:
            pass
    def redrawPlanar(self, iterations=200):
        self.graph.redraw_planar(iterations=iterations)
        self.updateGraphScene()
        # If you want auto center after redraws (not adds), uncomment:
        # self.animateCenterGraph(duration_ms=CENTER_ANIM_MS)
        try:
            stats = self.graph.get_stats()
            self.parent().statusBar().showMessage(
                f"Redraw complete. Total={stats['total_vertices']} "
                f"(seed={stats['seed']}, random={stats['random']}, manual={stats['manual']}), "
                f"periphery={stats['periphery_size']}",
                4000
            )
        except Exception:
            pass
        
    def declutterView(self, intensity: float = None):
        """Manual declutter with animation (planarity-safe)."""
        if intensity is None:
            intensity = getattr(self, "DECLUTTER_INTENSITY", 1.0)
        if not hasattr(self.graph, "declutter_for_view"):
            try:
                self.parent().statusBar().showMessage("Declutter not available.", 2000)
            except Exception:
                pass
            return
        pre = self._snapshot_positions()
        changed = False
        try:
            changed = bool(self.graph.declutter_for_view(float(intensity)))
        except Exception:
            changed = False
        if changed:
            post = self._snapshot_positions()
            self._animate_transition(pre, post, duration_ms=420)
            try:
                self.parent().statusBar().showMessage("Declutter complete.", 2000)
            except Exception:
                pass
        else:
            try:
                self.parent().statusBar().showMessage("Already well spaced.", 2000)
            except Exception:
                pass

    def setShrinkOnZoom(self, enabled: bool):
        """
        Toggle shrinking nodes as you zoom in.
        When disabled, gamma=1.0 (node px size ~ constant on screen).
        """
        if not hasattr(self, "_default_gamma"):
            self._default_gamma = getattr(self, "NODE_SHRINK_GAMMA", 1.20)
        self.NODE_SHRINK_GAMMA = (self._default_gamma if enabled else 1.0)
        self.updateGraphScene()

    def setNodeShrinkGamma(self, gamma: float):
        """
        Set the shrink strength gamma in [1.0..2.0]; >1 shrinks more as you zoom in.
        """
        g = max(1.0, min(2.0, float(gamma)))
        self.NODE_SHRINK_GAMMA = g
        self._default_gamma = g
        self.updateGraphScene()

    def _do_add_random_vertex(self):
        try:
            pre = self._snapshot_positions()
            ok, new_idx = self.graph.addRandomVertex()
            if not ok:
                # No valid arc; clear queue
                self._pending_random_adds = 0
                QMessageBox.warning(
                    self,
                    "No Valid Arc",
                    "No valid periphery arc found (degree ≥ 5 rule for hidden vertices)."
                )
                return

            # Animate the transition from pre -> post; spawn new vertex from its suggested spawn_pos
            post = self._snapshot_positions()
            info = self.graph.get_last_add_info()
            if info and info.get("index") is not None and info["index"] not in pre:
                pre[info["index"]] = info["spawn_pos"]

            self._animate_transition(pre, post, duration_ms=ADD_ANIM_MS)

            # UI updates
            try:
                self.parent().spinGoTo.setValue(len(self.graph.getVertices()))
                self.parent().statusBar().showMessage(f"Inserted vertex {new_idx + 1} (random).", 2000)
            except Exception:
                pass

        except RuntimeError as e:
            # Clear queue on error
            self._pending_random_adds = 0
            QMessageBox.warning(self, "No Valid Arc", str(e))
        except Exception as e:
            self._pending_random_adds = 0
            QMessageBox.critical(self, "Error", f"Unexpected failure: {e}")
        
    def _animate_transition(self, pre, post, duration_ms=ADD_ANIM_MS):
        # Stop camera anim if running to avoid blending
        if self._cameraTimeline and self._cameraTimeline.state() == QTimeLine.Running:
            self._cameraTimeline.stop()
        if self._highlightTimeline and self._highlightTimeline.state() == QTimeLine.Running:
            self._highlightTimeline.stop()

        # Start from 'pre'
        self._apply_positions_snapshot(pre)
        self.updateGraphScene()

        tl = self._makeTimeline(duration_ms)
        tl.valueChanged.connect(lambda t: self._apply_interpolated_positions(pre, post, t))

        def finish():
            # Snap to final layout
            self._apply_positions_snapshot(post)
            self.updateGraphScene()
            # Smoothly center to the exact view that centerGraph() would produce
            self.animateCenterGraph(duration_ms=CENTER_ANIM_MS)

            # When the camera animation settles, continue processing queued random adds
            if self._cameraTimeline and self._cameraTimeline.state() == QTimeLine.Running:
                if not self._waiting_for_camera_finish:
                    self._cameraTimeline.finished.connect(self._on_camera_anim_finished)
                    self._waiting_for_camera_finish = True
            else:
                # No camera animation -> proceed immediately
                self._attempt_process_random_queue()

        tl.finished.connect(finish)
        tl.start()
        self._animTimeline = tl

    # ---------- Export ----------
    def exportAsImage(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export as Image", "", "PNG Files (*.png);;SVG Files (*.svg)")
        if not path:
            return

        if self.scene().sceneRect().isEmpty():
            self.scene().setSceneRect(self.scene().itemsBoundingRect())

        if path.lower().endswith(".svg"):
            try:
                from PyQt5.QtSvg import QSvgGenerator
            except Exception:
                QMessageBox.warning(self, "SVG Export", "QtSvg module not available. Please export PNG.")
                return
            generator = QSvgGenerator()
            generator.setFileName(path)
            rect = self.scene().itemsBoundingRect().adjusted(-10, -10, 10, 10)
            size = rect.size().toSize()
            if size.width() <= 0 or size.height() <= 0:
                QMessageBox.warning(self, "Export", "Scene rect is empty; cannot export.")
                return
            generator.setSize(size)
            generator.setViewBox(rect)
            painter = QPainter()
            if painter.begin(generator):
                self.scene().render(painter, target=QRectF(0, 0, size.width(), size.height()), source=rect)
                painter.end()
        else:
            rect = self.scene().itemsBoundingRect().adjusted(-10, -10, 10, 10)
            size = rect.size().toSize()
            w = max(1, size.width())
            h = max(1, size.height())

            MAX_DIM = 12000
            scale = min(1.0, MAX_DIM / max(w, h))
            w2 = max(1, int(w * scale))
            h2 = max(1, int(h * scale))

            image = QImage(w2, h2, QImage.Format_ARGB32_Premultiplied)
            image.fill(Qt.transparent)

            painter = QPainter(image)
            if painter.isActive():
                self.scene().render(painter, target=QRectF(0, 0, w2, h2), source=rect)
                painter.end()
                if not image.save(path):
                    QMessageBox.warning(self, "Export", "Failed to save the PNG image.")
            else:
                from PyQt5.QtGui import QPixmap
                pix = QPixmap(w2, h2)
                pix.fill(Qt.transparent)
                p2 = QPainter(pix)
                if p2.isActive():
                    self.scene().render(p2, target=QRectF(0, 0, w2, h2), source=rect)
                    p2.end()
                    if not pix.save(path):
                        QMessageBox.warning(self, "Export", "Failed to save the PNG image (pixmap fallback).")
                else:
                    QMessageBox.warning(self, "Export", "Failed to start painter for image export. Try smaller image or SVG.")

    # ---------- Edge width ----------
    def _edgeWidthPx(self):
        zoom = max(0.0001, self.transform().m11())
        V = len(self.graph.vertices)
        return max(0.25, 2.0 / (1.0 + 0.9 * zoom + 0.00012 * V))

    def _scaleNow(self):
        return max(1e-9, self.transform().m11())

    def _minAllowedScale(self):
        base = self._base_fit_scale if self._base_fit_scale is not None else self._scaleNow()
        return max(MIN_ABS_SCALE, base * MIN_REL_TO_FIT)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, self.gridBackground)

    # ---------- Helpers (curved code retained but not used in strict mode) ----------
    def _periphery_is_ccw(self) -> bool:
        peri = self.graph.getPeriphery()
        if len(peri) < 3:
            return True
        pts = [self.graph.vertices[i].getPosition() for i in peri]
        area2 = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area2 += pts[i].x() * pts[j].y() - pts[j].x() * pts[i].y()
        return area2 > 0.0

    def _outward_normal_for_edge(self, p1: QPointF, p2: QPointF, orient_ccw: bool) -> QPointF:
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return QPointF(0.0, 0.0)
        if orient_ccw:
            nx, ny = dy / L, -dx / L
        else:
            nx, ny = -dy / L, dx / L
        return QPointF(nx, ny)

    def _sagitta_for_angle(self, chord_len: float, degrees: float) -> float:
        if degrees <= 0.0 or chord_len <= 0.0:
            return 0.0
        theta = math.radians(degrees)
        s_denom = 2.0 * math.sin(theta / 2.0)
        if abs(s_denom) < 1e-9:
            return 0.0
        return chord_len * (1.0 - math.cos(theta / 2.0)) / s_denom

    def _fixed_quadratic_control_in_face(self, u: int, v: int, p1: QPointF, p2: QPointF, degrees: float) -> Optional[QPointF]:
        au = self.graph._adjacency.get(u, set())
        av = self.graph._adjacency.get(v, set())
        thirds = list(au.intersection(av))
        if not thirds:
            return None

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return None
        n = QPointF(-dy / L, dx / L)
        mx, my = (p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5

        best_pw = None
        best_h = 0.0
        best_sgn = 1.0
        for w in thirds:
            pw = self.graph.vertices[w].getPosition()
            signed = (dx * (pw.y() - p1.y()) - dy * (pw.x() - p1.x())) / L
            h = abs(signed)
            sgn = 1.0 if signed > 0 else -1.0
            if h > best_h:
                best_h = h
                best_sgn = sgn
                best_pw = pw
        if best_pw is None:
            return None

        s_target = self._sagitta_for_angle(L, degrees)
        s = min(s_target, 0.45 * best_h, 0.20 * L)
        ctrl = QPointF(mx + best_sgn * n.x() * (2.0 * s), my + best_sgn * n.y() * (2.0 * s))

        tries = 0
        while tries < 10 and not self._point_in_triangle(ctrl, p1, p2, best_pw):
            s *= 0.7
            ctrl = QPointF(mx + best_sgn * n.x() * (2.0 * s), my + best_sgn * n.y() * (2.0 * s))
            tries += 1

        if not self._point_in_triangle(ctrl, p1, p2, best_pw):
            ctrl = QPointF((p1.x() + p2.x() + best_pw.x()) / 3.0, (p1.y() + p2.y() + best_pw.y()) / 3.0)

        return ctrl

    def _point_in_triangle(self, p: QPointF, a: QPointF, b: QPointF, c: QPointF) -> bool:
        def sign(p1, p2, p3):
            return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y())
        b1 = sign(p, a, b) < 0.0
        b2 = sign(p, b, c) < 0.0
        b3 = sign(p, c, a) < 0.0
        return (b1 == b2) and (b2 == b3)

    # ---------- Build star order (kept for potential future needs) ----------
    def _compute_star_geometry(self, visible_only=True):
        V = len(self.graph.vertices)
        star_order = {}
        ang_map = {}
        for u in range(V):
            vtx = self.graph.vertices[u]
            if not vtx or (visible_only and not vtx.isVisible()):
                continue
            pu = vtx.getPosition()
            neigh = list(self.graph._adjacency.get(u, set()))
            pairs = []
            for nb in neigh:
                nb_v = self.graph.vertices[nb]
                if not nb_v or (visible_only and not nb_v.isVisible()):
                    continue
                pv = nb_v.getPosition()
                ang = math.atan2(pv.y() - pu.y(), pv.x() - pu.x())
                pairs.append((ang, nb))
            pairs.sort(key=lambda t: t[0])
            order = [nb for ang, nb in pairs]
            star_order[u] = order
            for ang, nb in pairs:
                ang_map[(u, nb)] = ang
        return star_order, ang_map

    def updateGraphScene(self):
        if self._in_update_scene:
            return
        self._in_update_scene = True
        try:
            self.scene().clear()

            edge_width = self._edgeWidthPx()
            edge_pen = QPen(self.edgeColor)
            edge_pen.setWidthF(edge_width)
            edge_pen.setCosmetic(True)
            edge_pen.setCapStyle(Qt.RoundCap)
            edge_pen.setJoinStyle(Qt.RoundJoin)

            all_visible_vertices = [v for v in self.graph.getVertices() if v and v.isVisible()]
            if not all_visible_vertices:
                self.scene().setSceneRect(self.scene().itemsBoundingRect())
                self.viewport().update()
                return

            V = len(self.graph.vertices)
            aggregate_edges = (V >= self.AGGREGATE_EDGES_FROM)

            # --- Edges ---
            if aggregate_edges:
                path_all = QPainterPath()
                for edge in self.graph.getEdges():
                    if not edge.isVisible():
                        continue
                    p1 = edge.getStartVertex().getPosition()
                    p2 = edge.getEndVertex().getPosition()
                    path_all.moveTo(p1)
                    path_all.lineTo(p2)
                self.scene().addPath(path_all, edge_pen).setZValue(-10)
            else:
                P = self.graph.periphery
                peri = P.getIndices()
                n_peri = len(peri)
                orient_ccw = self._periphery_is_ccw()

                for edge in self.graph.getEdges():
                    if not edge.isVisible():
                        continue

                    start_v = edge.getStartVertex()
                    end_v = edge.getEndVertex()
                    u, v = start_v.getIndex(), end_v.getIndex()
                    p1, p2 = start_v.getPosition(), end_v.getPosition()

                    path = QPainterPath(p1)
                    control_point = None

                    if n_peri >= 2:
                        iu = P.indexOf(u)
                        is_periphery_edge = False
                        if iu >= 0:
                            is_periphery_edge = (
                                peri[(iu + 1) % n_peri] == v or
                                peri[(iu - 1) % n_peri] == v
                            )

                        if is_periphery_edge and self.curvedPeripheryEnabled:
                            is_uv_order = (peri[(iu + 1) % n_peri] == v) if iu >= 0 and n_peri > 0 else True
                            p1d = p1 if is_uv_order else p2
                            p2d = p2 if is_uv_order else p1
                            chord_len = v_len(v_sub(p2d, p1d))
                            mid = v_mid(p1d, p2d)
                            dx = p2d.x() - p1d.x()
                            dy = p2d.y() - p1d.y()
                            L = math.hypot(dx, dy)
                            if L > 1e-9:
                                nrm = QPointF(dy / L, -dx / L) if orient_ccw else QPointF(-dy / L, dx / L)
                                theta = math.radians(PERIPHERY_CURVE_DEG)
                                sag = 0.0 if theta <= 0 else chord_len * (1.0 - math.cos(theta / 2.0)) / (2.0 * math.sin(theta / 2.0))
                                control_point = v_add(mid, v_scale(nrm, 2.0 * sag))

                    if control_point:
                        path.quadTo(control_point, p2)
                    else:
                        path.lineTo(p2)

                    self.scene().addPath(path, edge_pen).setZValue(-10)

            # --- Nodes + Labels (zoom-aware node size; label = 30% of node) ---
            label_mode = self.graph.get_label_mode()
            scale_now = max(1e-6, self.transform().m11())

            for v in all_visible_vertices:
                base_d = v.getDiameter()          # scene-units baseline
                node_px = self._node_pixel_diameter(base_d, scale_now)  # on-screen px
                d_scene = node_px / scale_now     # convert to scene units for ellipse

                pos = v.getPosition()

                if label_mode == 1:
                    pen = QPen(QColor("#000000"), 2); pen.setCosmetic(True)
                    fill_color = QColor("#ffffff")
                else:
                    color_idx = (v.getColorIndex() - 1) % 4
                    pen = QPen(self.palette_outline[color_idx], 2); pen.setCosmetic(True)
                    fill_color = self.palette_fill[color_idx]

                self.scene().addEllipse(pos.x() - d_scene / 2, pos.y() - d_scene / 2, d_scene, d_scene, pen, fill_color).setZValue(10)

                # Index label sized to 30% of node diameter (on-screen) and centered
                if label_mode in (1, 2):
                    label = str(v.getIndex() + 1)
                    target_px = max(1, int(round(self.LABEL_NODE_FRACTION * node_px)))

                    font = QFont("Arial")
                    font.setPixelSize(target_px)

                    text = QGraphicsSimpleTextItem(label)
                    text.setFont(font)
                    text.setBrush(QColor("#000000") if label_mode == 2 else self.labelIndexOnlyColor)
                    text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

                    fm = QFontMetricsF(font)
                    w = fm.horizontalAdvance(label)
                    h = fm.ascent() + fm.descent()

                    dx_scene = (w * 0.5) / scale_now
                    dy_scene = (h * 0.5) / scale_now
                    text.setPos(pos.x() - dx_scene, pos.y() - dy_scene)
                    text.setZValue(20)
                    self.scene().addItem(text)

            # Fit scene rect (no view change)
            br = self.scene().itemsBoundingRect()
            if not br.isEmpty():
                self.scene().setSceneRect(br.adjusted(-50, -50, 50, 50))

            self.viewport().update()
        finally:
            self._in_update_scene = False

    def goToVertex(self, m):
        if not self.graph.getVertices() or m > len(self.graph.getVertices()):
            try:
                self.parent().spinGoTo.setValue(len(self.graph.getVertices()))
            except Exception:
                pass
            return
        self.graph.goToVertex(m)
        self.updateGraphScene()

    def zoomIn(self):
        scale_now = self._scaleNow()
        target = min(ZOOM_MAX, scale_now * ZOOM_FACTOR)
        if target <= scale_now + 1e-12:
            return
        factor = target / scale_now
        self.scale(factor, factor)
        self.currentZoom = self.transform().m11()
        self.viewport().update()
        self.updateGraphScene()
        self._maybeDeclutterAfterZoom()

    def zoomOut(self):
        scale_now = self._scaleNow()
        min_scale = self._minAllowedScale()
        target = scale_now / ZOOM_FACTOR
        if target <= min_scale + 1e-12:
            return
        factor = target / scale_now
        self.scale(factor, factor)
        self.currentZoom = self.transform().m11()
        self.viewport().update()
        self.updateGraphScene()
        # usually no declutter when zooming out; keep it optional
        # self._maybeDeclutterAfterZoom()

    def centerGraph(self):
        if self.scene().items():
            rect = self.scene().itemsBoundingRect()
            adjust = max(50, len(self.graph.vertices) / 100)
            safe_rect = rect.adjusted(-adjust, -adjust, adjust, adjust)
            if safe_rect.width() < 1e-6 or safe_rect.height() < 1e-6:
                return
            self.fitInView(safe_rect, Qt.KeepAspectRatio)
            self.currentZoom = self.transform().m11()
            self._base_fit_scale = self.currentZoom
            self.viewport().update()
            self.updateGraphScene()

    def smoothCenterGraph(self, duration_ms=CENTER_ANIM_MS):
        if not self.scene().items():
            return
        rect = self.scene().itemsBoundingRect()
        adjust = max(50, len(self.graph.vertices) / 100)
        target_rect = rect.adjusted(-adjust, -adjust, adjust, adjust)
        self._animateToSceneRect(target_rect, duration_ms)

    def toggleVertexDisplay(self):
        mode = self.graph.cycle_label_mode()
        try:
            name = ["Color", "Index", "Color + Index"][mode]
            self.parent().statusBar().showMessage(f"Label mode: {name}", 2000)
        except Exception:
            pass
        self.updateGraphScene()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoomIn()
        else:
            self.zoomOut()
        event.accept()


    def mousePressEvent(self, event):
        # Interactive periphery selection: click Vp then Vq
        if self.selectingPeriphery:
            # Allow right-click to cancel
            if event.button() == Qt.RightButton:
                self.selectingPeriphery = False
                self.peripheryStartIndex = -1
                self.setCursor(Qt.ArrowCursor)
                try:
                    self.parent().statusBar().showMessage("Selection cancelled.", 2000)
                except Exception:
                    pass
                return

            scenePos = self.mapToScene(event.pos())
            vIdx = self.findVertexAtPosition(scenePos)
            if vIdx == -1:
                # Clicked empty space -> cancel selection
                self.selectingPeriphery = False
                self.peripheryStartIndex = -1
                self.setCursor(Qt.ArrowCursor)
                try:
                    self.parent().statusBar().showMessage("Selection cancelled.", 2000)
                except Exception:
                    pass
                return

            if vIdx not in self.graph.getPeriphery():
                try:
                    self.parent().statusBar().showMessage(f"Vertex {vIdx + 1} is not on the periphery.", 3000)
                except Exception:
                    pass
                return

            if self.peripheryStartIndex == -1:
                # First endpoint (Vp)
                self.peripheryStartIndex = vIdx

                # Hint: which choice gives short arc vs full cycle (in current mode)
                try:
                    cw_nb, ccw_nb = self._cw_neighbors_hint(vIdx)
                    mode_name = "Visual" if self.selectionCwMode == 'visual' else "Program"
                    if cw_nb is not None and ccw_nb is not None:
                        self.parent().statusBar().showMessage(
                            f"Vp={vIdx + 1} selected [{mode_name} CW]. Short arc: pick {cw_nb + 1}. Full cycle: pick {ccw_nb + 1}.",
                            6000
                        )
                    else:
                        self.parent().statusBar().showMessage(
                            f"Vp={vIdx + 1} selected. Now select the second vertex (Vq).", 6000
                        )
                except Exception:
                    pass
                return

            if vIdx == self.peripheryStartIndex:
                # Same vertex picked twice — ignore
                try:
                    self.parent().statusBar().showMessage("Please pick a different Vq.", 2000)
                except Exception:
                    pass
                return

            # Second endpoint (Vq) -> perform insertion via (a, b), mapped to program CW if needed
            peripheryEndIndex = vIdx
            pre = self._snapshot_positions()

            a = self.peripheryStartIndex
            b = peripheryEndIndex
            a, b = self._map_selection_to_program_cw(a, b)

            ok, new_idx = self.graph.addVertexBySelection(a, b)

            # Reset selection UI state
            self.selectingPeriphery = False
            self.peripheryStartIndex = -1
            self.setCursor(Qt.ArrowCursor)

            if ok:
                try:
                    self.parent().spinGoTo.setValue(len(self.graph.getVertices()))
                except Exception:
                    pass
                post = self._snapshot_positions()
                info = self.graph.get_last_add_info()
                if info and info.get("index") is not None and info["index"] not in pre:
                    pre[info["index"]] = info["spawn_pos"]
                self._animate_transition(pre, post, duration_ms=ADD_ANIM_MS)
                try:
                    self.parent().statusBar().showMessage(f"New vertex {new_idx + 1} added by selection.", 3000)
                except Exception:
                    pass
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Insertion",
                    "Cannot insert here (degree ≥ 5 rule for hidden vertices, or invalid periphery arc)."
                )
            return

        # Default behavior (panning etc.)
        if event.button() == Qt.LeftButton:
            self.panning = True
            self.lastPanPoint = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = self.mapToScene(self.lastPanPoint) - self.mapToScene(event.pos())
            self.lastPanPoint = event.pos()
            self.translate(delta.x(), delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        act_add_random = menu.addAction("Add Random Vertex (R)")
        act_add_random.triggered.connect(self.addRandomVertex)
        act_add_random.setEnabled(not getattr(self, "finalizeMode", False))

        act_add_selection = menu.addAction("Add Vertex by Selection (A)")
        act_add_selection.triggered.connect(self.addVertexBySelection)
        act_add_selection.setEnabled(not getattr(self, "finalizeMode", False))

        menu.addSeparator()
        menu.addAction("Redraw (Planar) (D)", self.redrawPlanar)
        menu.addAction("Toggle Labels (T)", self.toggleVertexDisplay)
        menu.addAction("Center Graph (C)", self.centerGraph)

        menu.exec_(event.globalPos())

    def findVertexAtPosition(self, scenePos):
        for v in reversed(self.graph.getVertices()):
            if not v or not v.isVisible():
                continue
            dx = v.getPosition().x() - scenePos.x()
            dy = v.getPosition().y() - scenePos.y()
            dist = math.hypot(dx, dy)
            if dist <= v.getDiameter() / 2:
                return v.getIndex()
        return -1
    
    def cancelSelection(self):
        if self.selectingPeriphery:
            self.selectingPeriphery = False
            self.peripheryStartIndex = -1
            self.setCursor(Qt.ArrowCursor)
            try:
                self.parent().statusBar().showMessage("Selection cancelled.", 2000)
            except Exception:
                pass
        
    def _isBusyAnimating(self) -> bool:
        """True while vertex animation or camera centering is running."""
        anim_busy = (self._animTimeline and self._animTimeline.state() == QTimeLine.Running)
        cam_busy = (self._cameraTimeline and self._cameraTimeline.state() == QTimeLine.Running)
        return bool(anim_busy or cam_busy)

    def _on_camera_anim_finished(self):
        """Called when camera finishes centering; proceed with queued random adds."""
        self._waiting_for_camera_finish = False
        self._attempt_process_random_queue()

    def _attempt_process_random_queue(self):
        """
        If no animation is running and there are queued random-add requests,
        perform exactly one random add (then wait for its animation to settle).
        """
        if self._isBusyAnimating():
            return
        if self._pending_random_adds <= 0:
            return

        # Consume one request and perform it
        self._pending_random_adds -= 1
        self._do_add_random_vertex()
        

    def setTheme(self, mode: str):
        """
        Apply theme to the canvas (background, edge color, label tint for index-only mode).
        mode: 'light' or 'dark'
        """
        m = (mode or '').strip().lower()
        if m == 'dark':
            self.themeMode = 'dark'
            self.gridBackground = QColor(18, 18, 18)    # canvas background
            self.edgeColor = QColor(185, 185, 185)      # lighter edges for dark bg
            self.labelIndexOnlyColor = QColor(0, 0, 0)  # black text on white node (index-only mode)
        else:
            self.themeMode = 'light'
            self.gridBackground = QColor(255, 255, 255)
            self.edgeColor = QColor(60, 60, 60)
            self.labelIndexOnlyColor = QColor(0, 0, 0)

        self.updateGraphScene()