import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum

from .poly import Poly
from .helper import unwrap

class Mode(Enum):
  DRAW = 1
  DEFORM_TRANS = 2
  DEFORM_RIGID = 3
  DEFORM_SIMILAR = 5
  DEFORM_AFFINE = 6
  DEFORM_PERSPECTIVE = 7

@dataclass()
class State:
  cursor: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  action_start: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  mode: Mode = Mode.DRAW
  mb_down: bool = False
  selected: tuple[Poly, int] | None = None

class Window:
  def __init__(self, name: str):
    self.win = name
    self.state = State()
    self.should_close = False
    self.polygons: list[Poly] = []
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 900, 900)
    cv.setMouseCallback(name, self._on_mouse)
    
  def __del__(self):
    cv.destroyWindow(self.win)
    
  def _is_open(self) -> bool:
    try:
      return cv.getWindowProperty(self.win, cv.WND_PROP_VISIBLE) >= 1
    except cv.error:
      return False
   
  def render(self) -> bool:
    img = np.full((900, 900, 3), 10, np.uint8)
    self._wait_key()
    
    if not self._is_open() or self.should_close:
      return False
    
    self._draw_keybinds(img)
     
    for poly in self.polygons:
      cv.polylines(img, [poly.pts], True, (0, 255, 255))
      for i, p in enumerate(poly.pts):
        color = (255, 0, 0) if (self.state.selected != None and self.state.selected[0] is poly and self.state.selected[1] == i) else (0, 255, 0)
        cv.circle(img, p, 5, color, cv.FILLED)
        
    cv.imshow(self.win, img)
    
    return True
  
  def _draw_keybinds(self, img):
    cv.putText(img, "Draw mode: d", (50, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform translation mode: t", (50, 75), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform rigid mode: r", (50, 100), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform similarity mode: s", (50, 125), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform affine mode: a", (50, 150), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform perspective mode: p", (50, 175), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
  
  def _on_mouse(self, event, x, y, flags, _: Any):
    self.state.cursor = np.array([x, y])
    
    match event:
      case cv.EVENT_LBUTTONDOWN:
        self._on_mb_down()
      case cv.EVENT_LBUTTONUP:
        self._on_mb_up()
      case cv.EVENT_MOUSEMOVE:
        self._on_mouse_move()
          
  def _on_mb_down(self):
    self.state.mb_down = True
    self.state.action_start = self.state.cursor.copy()
    
    if self.state.mode != Mode.DRAW:
      self._select_point()
    else: 
      self.state.mode = Mode.DRAW
      self.polygons.append(Poly(np.tile(self.state.cursor, (4, 1))))
      self.state.selected = (self.polygons[-1], 2)
      
  def _on_mb_up(self):
    self.state.mb_down = False
    if (sel := self.state.selected): sel[0].set_origin()
    self.state.selected = None
      
  def _on_mouse_move(self):
    if not self.state.mb_down: return
    x, y = self.state.cursor
    x0, y0 = self.state.action_start
    dx, dy = self.state.cursor - self.state.action_start
    match self.state.mode, self.state.selected:
      case Mode.DRAW, (poly, _):
        poly.pts = np.array([[x0, y0], [x0, y], [x, y], [x, y0]])
      case Mode.DEFORM_TRANS, (poly, _):
        poly.pts[:] = poly.origin_pts + [dx, dy]
        #T = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
        #self.state.selected_poly.pts[:] = [(T @ np.r_[p, 1])[:2] for p in self.state.selected_poly.origin_pts]
      case Mode.DEFORM_RIGID, (poly, pt):
        center = np.mean(poly.origin_pts, axis=0)
        r = self.state.action_start - center
        v = self.state.cursor - center
        if np.linalg.norm(r) > 1e-8 and np.linalg.norm(v) > 1e-8:
          rot_angle = np.arctan2(r[0] * v[1] - r[1] * v[0], np.dot(r, v))
          s, c = np.sin(rot_angle), np.cos(rot_angle)
          R = np.array([[c, -s],[s,  c]])
          poly.pts[:] = (poly.origin_pts - center) @ R.T + center
        
      case Mode.DEFORM_SIMILAR: pass
      case Mode.DEFORM_AFFINE: pass
      case Mode.DEFORM_PERSPECTIVE, (poly, pt):
        poly.pts[pt] = self.state.cursor
  
  def _wait_key(self):
    key = cv.waitKey(16)
    if key == ord('q'): self.should_close = True
    elif key == ord('d'): self.state.mode = Mode.DRAW
    elif key == ord('t'): self.state.mode = Mode.DEFORM_TRANS
    elif key == ord('r'): self.state.mode = Mode.DEFORM_RIGID
    elif key == ord('s'): self.state.mode = Mode.DEFORM_SIMILAR
    elif key == ord('a'): self.state.mode = Mode.DEFORM_AFFINE
    elif key == ord('p'): self.state.mode = Mode.DEFORM_PERSPECTIVE
      
  def _select_point(self):
    for poly in self.polygons:
      for i, p in enumerate(poly.pts):
        if np.linalg.norm(p - self.state.cursor) <= 25:
          self.state.selected = (poly, i)
          return