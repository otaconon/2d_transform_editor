from collections.abc import Iterable
from dataclasses import dataclass, field
import numpy as np
from enum import Enum, auto
from copy import copy

class Mode(Enum):
  DRAW = auto()
  DEFORM_TRANS = auto()
  DEFORM_RIGID = auto()
  DEFORM_SIMILAR = auto()
  DEFORM_AFFINE = auto()
  DEFORM_PERSPECTIVE = auto()

@dataclass
class Action:
  mode: Mode = Mode.DRAW
  start: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  end: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  dr: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.int32))
  pt: int = 0

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
  
def rotation(u, v) -> np.ndarray:
  if (denom := np.hypot(*u) * np.hypot(*v)) <= 1e-8: return np.array([1,0],[0,1])
  c, s = (u @ v) / denom, (u[0] * v[1] - u[1] * v[0]) / denom
  return np.array([[c, -s], [s, c]])
  
class Poly:
  __slots__ = ("pts", "origin_pts")

  def __init__(self, pts: Iterable[Point] | np.ndarray) -> None:
    if isinstance(pts, np.ndarray): self.pts = np.ascontiguousarray(pts, dtype=np.int32)
    else: self.pts = np.array([p.arr for p in pts], dtype=np.int32)
    self.finish_action()
      
  def __getitem__(self, idx: int) -> Point:
    return Point(self.pts[idx])
  
  def finish_action(self):
    self.origin_pts = self.pts.copy()
    
  def deform(self, a: Action):
    if a.mode == Mode.DRAW: self.deform_draw(a)
    if a.mode == Mode.DEFORM_TRANS: self.deform_trans(a)
    if a.mode == Mode.DEFORM_RIGID: self.deform_rigid(a)
    if a.mode == Mode.DEFORM_SIMILAR: self.deform_similar(a)
    if a.mode == Mode.DEFORM_AFFINE: self.deform_affine(a)
    if a.mode == Mode.DEFORM_PERSPECTIVE: self.deform_perspective(a)
    
  def deform_draw(self, a: Action):
    (x0, y0), (x, y) = a.start, a.end
    self.pts = np.array([[x0, y0], [x0, y], [x, y], [x, y0]])
    
  def deform_trans(self, a: Action):
    self.pts[:] = self.origin_pts + a.dr
   
  def deform_rigid(self, a: Action):
    center = self.origin_pts.mean(axis=0)
    r, v = a.start - center, a.end - center
    R = rotation(r, v)
    self.pts[:] = np.rint((self.origin_pts - center) @ R.T + center)
    
  def deform_similar(self, a: Action):
    center = self.origin_pts.mean(axis=0)
    r, v = a.start - center, a.end - center
    R = rotation(r, v)
    scale = np.hypot(*v)/np.hypot(*r)
    self.pts[:] = np.rint((self.origin_pts - center) @ (R.T*scale) + center)
    
  def deform_affine(self, a: Action):
    self.deform_perspective(a)
    b = copy(a)
    b.dr = -a.dr
    b.pt = (a.pt+2)%len(self.pts)
    self.deform_perspective(b)
  
  def deform_perspective(self, a: Action):
    self.pts[a.pt] = self.origin_pts[a.pt] + a.dr