import cv2 as cv
import numpy as np
from typing import Any
from contextlib import suppress

from .poly import Poly, Mode, Action
  
MODES = {
  ord('d'): (Mode.DRAW, "Draw: d"),
  ord('t'): (Mode.DEFORM_TRANS, "Translation: t"),
  ord('r'): (Mode.DEFORM_RIGID, "Rigid: r"),
  ord('s'): (Mode.DEFORM_SIMILAR, "Similarity: s"),
  ord('a'): (Mode.DEFORM_AFFINE, "Affine: a"),
  ord('p'): (Mode.DEFORM_PERSPECTIVE, "Perspective: p"),
}

class Window:
  def __init__(self, name: str):
    self.name = name
    self.action = Action()
    self.polygons: list[Poly] = []
    self.mb_down: bool = False
    self.sel_poly: Poly | None = None
    
  def __enter__(self):
    cv.namedWindow(self.name, cv.WINDOW_NORMAL)
    cv.resizeWindow(self.name, 900, 900)
    cv.setMouseCallback(self.name, self._on_mouse)
    return self
    
  def __exit__(self, exc_type, exc_val, exc_tb):
    with suppress(cv.error):
      cv.destroyWindow(self.name)
    
  def render(self) -> bool:
    img = np.full((900, 900, 3), 10, np.uint8)
    for i, (mode, text) in enumerate(MODES.values()):
      color = (0, 255, 0) if mode == self.action.mode else (255, 255, 255)
      cv.putText(img, text, (50, 50+25*i), cv.FONT_HERSHEY_PLAIN, 1, color)
    cv.polylines(img, [poly.pts for poly in self.polygons], True, (0, 255, 255))
    for poly in self.polygons:
      for i, p in enumerate(poly.pts):
        color = (255, 0, 0) if (self.sel_poly != None and self.sel_poly is poly and self.action.pt == i) else (0, 255, 0)
        cv.circle(img, p, 5, color, cv.FILLED)
    cv.imshow(self.name, img)
    
    key = cv.waitKey(16)
    try:
      if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1 or key == ord('q'): return False
    except cv.error:
      return False
    if key in MODES: self.action.mode = MODES[key][0]
    
    return True
  
  def _on_mouse(self, event, x, y, flags, _: Any):
    self.action.end = np.array([x, y])
    self.action.dr = self.action.end - self.action.start
    
    mode = self.action.mode
    if (event, mode) == (cv.EVENT_LBUTTONDOWN, Mode.DRAW):
      self.polygons.append(Poly(np.tile((x, y), (4, 1))))
    if event == cv.EVENT_LBUTTONDOWN:
      self.mb_down, self.action.start = True, self.action.end.copy()
      self.sel_poly, self.action.pt = next(((poly, i) for poly in self.polygons for (i, p) in enumerate(poly.pts) if np.sum((p - self.action.end)**2) <= 625), (None, 0)) if mode != Mode.DRAW else (self.polygons[-1], 2)
    if event == cv.EVENT_LBUTTONUP and self.sel_poly is not None: self.sel_poly.record_state(self.action)
    if event == cv.EVENT_LBUTTONUP: self.mb_down, self.sel_poly = False, None
    if not self.mb_down or event != cv.EVENT_MOUSEMOVE: return
    if self.sel_poly == None: return
    
    if mode == Mode.DRAW: self.sel_poly.deform_draw(self.action)
    if mode == Mode.DEFORM_TRANS: self.sel_poly.deform_trans(self.action)
    if mode == Mode.DEFORM_RIGID: self.sel_poly.deform_rigid(self.action)
    if mode == Mode.DEFORM_SIMILAR: self.sel_poly.deform_similar(self.action)
    if mode == Mode.DEFORM_AFFINE: self.sel_poly.deform_affine(self.action)
    if mode == Mode.DEFORM_PERSPECTIVE: self.sel_poly.deform_perspective(self.action)