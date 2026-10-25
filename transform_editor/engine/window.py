import cv2 as cv
import numpy as np
from typing import Any
from contextlib import suppress
from copy import copy

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
  def __init__(self, name: str, width: int, height: int):
    self.name = name
    self.window_size = (width, height)
    self.action = Action(Mode.DRAW)
    self.action_record: list[tuple[Poly, Mode, np.ndarray]] = []
    self.polygons: list[Poly] = []
    self.mb_down: bool = False
    self.sel_poly: Poly | None = None
    
  def __enter__(self):
    cv.namedWindow(self.name, cv.WINDOW_NORMAL)
    cv.resizeWindow(self.name, *self.window_size)
    cv.setMouseCallback(self.name, self._on_mouse)
    return self
    
  def __exit__(self, exc_type, exc_val, exc_tb):
    with suppress(cv.error):
      cv.destroyWindow(self.name)
    
  def render(self) -> bool:
    img = np.full((*self.window_size, 3), 10, np.uint8)
    for i, (mode, text) in enumerate(MODES.values()):
      color = (0, 255, 0) if mode == self.action.mode else (255, 255, 255)
      cv.putText(img, text, (50, 50+25*i), cv.FONT_HERSHEY_PLAIN, 1, color)
    cv.putText(img, "Undo: u", (50, 50+25*len(MODES.values())), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
      
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
    if key == ord('u') and len(self.action_record)>0:
      poly, mode, pts = self.action_record.pop()
      if mode == Mode.DRAW:
        self.polygons.remove(poly)
      else:
        poly.pts = pts.copy()
        poly.finish_action()
    
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
    if event == cv.EVENT_LBUTTONUP and self.sel_poly is not None:
      self.action_record.append((self.sel_poly, self.action.mode, self.sel_poly.origin_pts.copy()))
      self.sel_poly.finish_action()
      self.sel_poly = None
    if event == cv.EVENT_LBUTTONUP: self.mb_down = False
    if not self.mb_down or event != cv.EVENT_MOUSEMOVE: return
    if self.sel_poly == None: return
    
    self.sel_poly.deform(self.action) 