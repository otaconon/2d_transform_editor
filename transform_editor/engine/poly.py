from collections.abc import Iterable
import numpy as np

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
  __slots__ = ("pts",)

  def __init__(self, pts: Iterable[Point] | np.ndarray) -> None:
    if isinstance(pts, np.ndarray):
      self.pts = np.ascontiguousarray(pts, dtype=np.int32)
    else:
      self.pts = np.array([p.arr for p in pts], dtype=np.int32)

  def __getitem__(self, idx: int) -> Point:
    return Point(self.pts[idx])