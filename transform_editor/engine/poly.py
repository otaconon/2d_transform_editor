from collections.abc import Iterable
from dataclasses import dataclass, field
import numpy as np
from enum import Enum, auto

class Mode(Enum):
  DRAW = auto()
  DEFORM_TRANS = auto()
  DEFORM_RIGID = auto()
  DEFORM_SIMILAR = auto()
  DEFORM_AFFINE = auto()
  DEFORM_PERSPECTIVE = auto()

@dataclass
class State:
  mode: Mode = Mode.DRAW
  selected: tuple[Poly, int] | None = None
  action_start: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))

class Point:
  __slots__ = ("arr",)

  def __init__(self, x: int, y: int | None = None) -> None:
    self.arr = np.array([x, y], dtype=np.int32)

  @property
  def x(self) -> int:
    return int(self.arr[0])

  @x.setter
  def x(self, val: int) -> None:
    self.arr[0] = val

  @property
  def y(self) -> int:
    return int(self.arr[1])

  @y.setter
  def y(self, val: int) -> None:
    self.arr[1] = val

  def __array__(self, dtype=np.int32, copy=None) -> np.ndarray:
    return self.arr.astype(dtype, copy=False)

class Poly:
  __slots__ = ("pts", "origin_pts", "state_record")

  def __init__(self, pts: Iterable[Point] | np.ndarray) -> None:
    if isinstance(pts, np.ndarray): self.pts = np.ascontiguousarray(pts, dtype=np.int32)
    else: self.pts = np.array([p.arr for p in pts], dtype=np.int32)
    self.state_record = []
      
  def __getitem__(self, idx: int) -> Point:
    return Point(self.pts[idx])
  
  def record_state(self, state: State):
    self.state_record.append(state) 
    self.origin_pts = self.pts.copy()
    
  def translate(self, t):
    self.pts[:] = self.origin_pts + t