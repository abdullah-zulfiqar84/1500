class QPointF:
    __slots__ = ("_x", "_y")
    def __init__(self, x=0.0, y=0.0):
        self._x = float(x); self._y = float(y)
    def x(self): return self._x
    def y(self): return self._y
    def __repr__(self): return f"QPointF({self._x:.3f}, {self._y:.3f})"