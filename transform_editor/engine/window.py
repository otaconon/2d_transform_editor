import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum, auto

from .poly import Poly
from .helper import unwrap

class Mode(Enum):
  DRAW = auto()
  DEFORM_TRANS = auto()
  DEFORM_RIGID = auto()
  DEFORM_SIMILAR = auto()
  DEFORM_AFFINE = auto()
  DEFORM_PERSPECTIVE = auto()
  
MODES = {
  ord('d'): (Mode.DRAW, "Draw: d"),
  ord('t'): (Mode.DEFORM_TRANS, "Translation: t"),
  ord('r'): (Mode.DEFORM_RIGID, "Rigid: r"),
  ord('s'): (Mode.DEFORM_SIMILAR, "Similarity: s"),
  ord('a'): (Mode.DEFORM_AFFINE, "Affine: a"),
  ord('p'): (Mode.DEFORM_PERSPECTIVE, "Perspective: p"),
}

@dataclass()
class State:
  cursor: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  action_start: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  mode: Mode = Mode.DRAW
  mb_down: bool = False
  selected: tuple[Poly, int] | None = None

class Window:
  def __init__(self, name: str):
    self.name = name
    self.state = State()
    self.should_close = False
    self.polygons: list[Poly] = []
    
  def __enter__(self):
    cv.namedWindow(self.name, cv.WINDOW_NORMAL)
    cv.resizeWindow(self.name, 900, 900)
    cv.setMouseCallback(self.name, self._on_mouse)
    return self
    
  def __exit__(self, exc_type, exc_val, exc_tb):
    cv.destroyWindow(self.name)
    
  def _is_open(self) -> bool:
    try:
      return cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) >= 1
    except cv.error:
      return False
   
  def render(self) -> bool:
    img = np.full((900, 900, 3), 10, np.uint8)
    
    key = cv.waitKey(16)
    if key == ord('q'): self.should_close = True
    elif key in MODES: self.state.mode = MODES[key][0]
    
    if not self._is_open() or self.should_close:
      return False
    
    for i, (_, text) in enumerate(MODES.values()):
      cv.putText(img, text, (50, 50+25*i), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
     
    cv.polylines(img, [poly.pts for poly in self.polygons], True, (0, 255, 255))
    for poly in self.polygons:
      for i, p in enumerate(poly.pts):
        color = (255, 0, 0) if (self.state.selected != None and self.state.selected[0] is poly and self.state.selected[1] == i) else (0, 255, 0)
        cv.circle(img, p, 5, color, cv.FILLED)
        
    cv.imshow(self.name, img)
    
    return True
  
  def _on_mouse(self, event, x, y, flags, _: Any):
    self.state.cursor = np.array([x, y])
    mode = self.state.mode
    
    if (event, mode) == (cv.EVENT_LBUTTONDOWN, Mode.DRAW):
      self.polygons.append(Poly(np.tile(self.state.cursor, (4, 1))))
    if event == cv.EVENT_LBUTTONDOWN:
      self.state.mb_down, self.state.action_start = True, self.state.cursor.copy()
      self.state.selected = next(((poly, i) for poly in self.polygons for (i, p) in enumerate(poly.pts) if np.sum((p - self.state.cursor)**2) <= 625), None) if mode != Mode.DRAW else (self.polygons[-1], 2)
    if event == cv.EVENT_LBUTTONUP and self.state.selected is not None: self.state.selected[0].set_origin()
    if event == cv.EVENT_LBUTTONUP: self.state.mb_down, self.state.selected = False, None
    
    if not self.state.mb_down or event != cv.EVENT_MOUSEMOVE: return
    x, y = self.state.cursor
    x0, y0 = self.state.action_start
    dx, dy = self.state.cursor - self.state.action_start
    if self.state.selected == None: return
    poly, pt = self.state.selected
    if mode == Mode.DRAW: poly.pts = np.array([[x0, y0], [x0, y], [x, y], [x, y0]])
    if mode == Mode.DEFORM_TRANS: poly.pts[:] = poly.origin_pts + [dx, dy]
      #T = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
      #self.state.selected_poly.pts[:] = [(T @ np.r_[p, 1])[:2] for p in self.state.selected_poly.origin_pts]
    if mode == Mode.DEFORM_RIGID:
      center = poly.origin_pts.mean(axis=0)
      r, v = self.state.action_start - center, self.state.cursor - center
      if (denom := np.hypot(*r) * np.hypot(*v)) <= 1e-8: return
      c, s = (r @ v) / denom, (r[0] * v[1] - r[1] * v[0]) / denom
      R = np.array([[c, -s], [s, c]])
      poly.pts[:] = np.rint((poly.origin_pts - center) @ R.T + center)
    if mode == Mode.DEFORM_SIMILAR: pass
    if mode == Mode.DEFORM_AFFINE: pass
    if mode == Mode.DEFORM_PERSPECTIVE: poly.pts[pt] = self.state.cursor