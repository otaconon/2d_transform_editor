import cv2 as cv
import numpy as np
from typing import Any
from contextlib import suppress

from .poly import Poly, Mode, State
  
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
    self.state = State()
    self.polygons: list[Poly] = []
    self.mb_down: bool = False
    
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
      color = (0, 255, 0) if mode == self.state.mode else (255, 255, 255)
      cv.putText(img, text, (50, 50+25*i), cv.FONT_HERSHEY_PLAIN, 1, color)
    cv.polylines(img, [poly.pts for poly in self.polygons], True, (0, 255, 255))
    for poly in self.polygons:
      for i, p in enumerate(poly.pts):
        color = (255, 0, 0) if (self.state.selected != None and self.state.selected[0] is poly and self.state.selected[1] == i) else (0, 255, 0)
        cv.circle(img, p, 5, color, cv.FILLED)
    cv.imshow(self.name, img)
    
    key = cv.waitKey(16)
    try:
      if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1 or key == ord('q'): return False
    except cv.error:
      return False
    if key in MODES: self.state.mode = MODES[key][0]
    
    return True
  
  def _on_mouse(self, event, x, y, flags, _: Any):
    cursor_pos = np.array([x,y])
    mode = self.state.mode
    
    if (event, mode) == (cv.EVENT_LBUTTONDOWN, Mode.DRAW):
      self.polygons.append(Poly(np.tile((x, y), (4, 1))))
    if event == cv.EVENT_LBUTTONDOWN:
      self.mb_down, self.state.action_start = True, cursor_pos.copy()
      self.state.selected = next(((poly, i) for poly in self.polygons for (i, p) in enumerate(poly.pts) if np.sum((p - cursor_pos)**2) <= 625), None) if mode != Mode.DRAW else (self.polygons[-1], 2)
    if event == cv.EVENT_LBUTTONUP and self.state.selected is not None: self.state.selected[0].record_state(self.state)
    if event == cv.EVENT_LBUTTONUP: self.mb_down, self.state.selected = False, None
    
    if not self.mb_down or event != cv.EVENT_MOUSEMOVE: return
    if self.state.selected == None: return
    
    poly, pt = self.state.selected
    dx, dy = cursor_pos - self.state.action_start
    dr = np.array([dx, dy])
    if mode == Mode.DRAW:
      x0, y0 = self.state.action_start
      poly.pts = np.array([[x0, y0], [x0, y], [x, y], [x, y0]])
    if mode == Mode.DEFORM_TRANS: poly.translate(dr)
    if mode == Mode.DEFORM_RIGID:
      center = poly.origin_pts.mean(axis=0)
      r, v = self.state.action_start - center, cursor_pos - center
      if (denom := np.hypot(*r) * np.hypot(*v)) <= 1e-8: return
      c, s = (r @ v) / denom, (r[0] * v[1] - r[1] * v[0]) / denom
      R = np.array([[c, -s], [s, c]])
      poly.pts[:] = np.rint((poly.origin_pts - center) @ R.T + center)
    if mode == Mode.DEFORM_SIMILAR:
      center = poly.origin_pts.mean(axis=0)
      r, v = self.state.action_start - center, cursor_pos - center
      if (denom := np.hypot(*r) * np.hypot(*v)) <= 1e-8: return
      cos, sin = (r @ v) / denom, (r[0] * v[1] - r[1] * v[0]) / denom
      scale = np.hypot(*v)/np.hypot(*r)
      R = np.array([[cos, -sin], [sin, cos]])*scale
      poly.pts[:] = np.rint((poly.origin_pts - center) @ R.T + center)
    if mode == Mode.DEFORM_AFFINE:
      poly.pts[pt] = poly.origin_pts[pt] + [dx, dy]
      poly.pts[(pt+2)%len(poly.pts)] = poly.origin_pts[(pt+2)%len(poly.pts)] - [dx, dy]
    if mode == Mode.DEFORM_PERSPECTIVE: poly.pts[pt] = cursor_pos