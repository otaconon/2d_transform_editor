import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum

from .poly import Poly

class Mode(Enum):
  DRAW = 1
  DEFORM_TRANS = 2
  DEFORM_RIGID = 3
  DEFORM_SIMILAR = 5
  DEFORM_AFFINE = 6
  DEFORM_PERSPECTIVE = 7

@dataclass()
class State:
  polygons: list[Poly] = field(default_factory=list)
  cursor: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  drawing_start: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  mode: Mode = Mode.DRAW
  mb_down: bool = False
  selected_pt: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

class Window:
  def __init__(self, name: str):
    self.win = name
    self.state = State()
    self.should_close = False
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
    if not self._is_open():
      return False
    
    self._draw_keybinds(img)
    
    for poly in self.state.polygons:
      cv.polylines(img, [poly.pts], True, (0, 255, 255))
      for p in poly.pts:
        if p is self.state.selected_pt:
          cv.circle(img, p, 5, (255, 0, 0), cv.FILLED)
        else:
          cv.circle(img, p, 5, (0, 255, 0), cv.FILLED)
      
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
        self.state.mb_down = False
      case cv.EVENT_MOUSEMOVE:
        self._on_mouse_move()
          
  def _on_mb_down(self):
    self.state.mb_down = True
    if self.state.mode != Mode.DRAW:
      self._select_point()
    else: 
      self.state.mode = Mode.DRAW
      self.state.drawing_start = self.state.cursor.copy()
      self.state.polygons.append(Poly(np.tile(self.state.cursor, (4, 1))))
      
  def _on_mouse_move(self):
    x, y = self.state.cursor
    x0, y0 = self.state.drawing_start
    if not self.state.mb_down: return
    match self.state.mode:
      case Mode.DRAW: self.state.polygons[-1].pts = np.array([[x0, y0], [x0, y,], [x, y], [x, y0]])
      case Mode.DEFORM_TRANS: pass
      case Mode.DEFORM_RIGID: pass
      case Mode.DEFORM_SIMILAR: pass
      case Mode.DEFORM_AFFINE: pass
      case Mode.DEFORM_PERSPECTIVE: self.state.selected_pt[:] = self.state.cursor
  
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
    for poly in self.state.polygons:
      for p in poly.pts:
        if np.linalg.norm(p - self.state.cursor) <= 25:
          self.state.selected_pt = p
          return