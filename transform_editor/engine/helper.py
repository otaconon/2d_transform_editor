from typing import TypeVar

T = TypeVar("T")

def unwrap(x: T | None) -> T:
  assert x is not None
  return x