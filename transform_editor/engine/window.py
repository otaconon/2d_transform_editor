import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Any, cast

from .poly import Poly

@dataclass()
class State:
  polygons: list[Poly] = field(default_factory=list)
  cursor: tuple[int, int] = (0, 0)
  drawing_start: tuple[int, int] = (0, 0)
  drawing: bool = False

class Window:
  def __init__(self, name: str):
    self.win = name
    self.state = State()
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(name, self._on_mouse)
    
  def __del__(self):
    cv.destroyWindow(self.win)
    
  def render(self) -> bool:
    img = np.full((600, 800, 3), 10, np.uint8)
    
    if cv.waitKey(16) & 0xFF == ord('q'):
      return False
    
    for poly in self.state.polygons:
      cv.polylines(img, [poly.pts], True, (0, 255, 255))
      for p in poly.pts:
        cv.circle(img, p, 5, (0, 255, 0), cv.FILLED)
      
    cv.imshow(self.win, img)
    
    if cv.getWindowProperty(self.win, cv.WND_PROP_VISIBLE) < 1:
      return False
    
    return True
      
  
  def _on_mouse(self, event, x, y, flags, _: Any):
    self.state.cursor = (x, y)
    x0, y0 = self.state.drawing_start
    
    if self.state.drawing:
      self.state.polygons[-1].pts = np.array([[x0, y0], [x0, y,], [x, y], [x, y0]])
      
    if event == cv.EVENT_LBUTTONDOWN:
      self.state.drawing = True
      self.state.drawing_start = (x, y)
      self.state.polygons.append(Poly(np.array([[x0, y0], [x0, y,], [x, y], [x, y0]])))
    elif event == cv.EVENT_LBUTTONUP and self.state.drawing:
      self.state.drawing = False