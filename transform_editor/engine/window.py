import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Any, cast

from .rect import Rect

@dataclass()
class State:
  shapes: list = field(default_factory=list)
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
    shapes = self.state.shapes
    img = np.full((600, 800, 3), 10, np.uint8)
    
    if cv.waitKey(16) & 0xFF == ord('q'):
      return False
    
    for s in shapes:
      self.draw_shape(img, s)
      
    cv.imshow(self.win, img)
    
    if cv.getWindowProperty(self.win, cv.WND_PROP_VISIBLE) < 1:
      return False
    
    return True
      
  def draw_shape(self, img, shape):
    if isinstance(shape, Rect):
      cv.rectangle(img, (shape.x, shape.y), (shape.w, shape.h), (0, 255, 255), cv.FILLED)
  
  def _on_mouse(self, event, x, y, flags, _: Any):
    self.state.cursor = (x, y)
    
    if self.state.drawing:
      self.state.shapes[-1] = Rect(*self.state.drawing_start, x, y)
      
    if event == cv.EVENT_LBUTTONDOWN:
      self.state.drawing = True
      self.state.drawing_start = (x, y)
      self.state.shapes.append(Rect(*self.state.drawing_start, x, y))
    elif event == cv.EVENT_LBUTTONUP and self.state.drawing:
      self.state.drawing = False