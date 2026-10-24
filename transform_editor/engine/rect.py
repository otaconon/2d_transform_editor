from dataclasses import dataclass

@dataclass(slots=True)
class Rect:
  x: int
  y: int
  w: int
  h: int