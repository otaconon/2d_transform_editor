import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum

from .poly import Poly

class Mode(Enum):
  IDLE = 1
  DRAW = 2
  DEFORM_TRANS = 3
  DEFORM_PROJ = 4

@dataclass()
class State:
  polygons: list[Poly] = field(default_factory=list)
  cursor: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  drawing_start: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  mode: Mode = Mode.IDLE
  selected_pt: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

class Window:
  def __init__(self, name: str):
    self.win = name
    self.state = State()
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
    
    if cv.waitKey(16) & 0xFF == ord('q'):
      return False
    
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
    cv.putText(img, "Deform translation mode: d", (50, 75), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform rigid mode: d", (50, 100), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform similarity mode: d", (50, 125), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform affine mode: d", (50, 150), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv.putText(img, "Deform perspective mode: d", (50, 175), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
  
  def _on_mouse(self, event, x, y, flags, _: Any):
    self.state.cursor = np.array([x, y])
    x0, y0 = self.state.drawing_start
    
    match event:
      case cv.EVENT_LBUTTONDOWN:
        if self.state.mode in (Mode.DEFORM_PROJ, Mode.DEFORM_TRANS):
          self._select_point()
        else: 
          self.state.mode = Mode.DRAW
          self.state.drawing_start = self.state.cursor.copy()
          self.state.polygons.append(Poly(np.tile(self.state.cursor, (4, 1))))
      case cv.EVENT_LBUTTONUP:
        self.state.mode = Mode.IDLE
      case cv.EVENT_MOUSEMOVE:
        if self.state.mode == Mode.DRAW: self.state.polygons[-1].pts = np.array([[x0, y0], [x0, y,], [x, y], [x, y0]])
        elif self.state.mode == Mode.DEFORM_PROJ: self.state.selected_pt[:] = self.state.cursor
      
  def _select_point(self):
    for poly in self.state.polygons:
      for p in poly.pts:
        if np.linalg.norm(p - self.state.cursor) <= 100:
          self.state.mode = Mode.DEFORM_PROJ
          self.state.selected_pt = p
          return