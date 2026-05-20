from dataclasses import dataclass
from typing import Any, Self
import math
import numpy as np

@dataclass(frozen = True, slots = True)
class Vec3f:
    """ 3-component floating-point vector """
    x: float
    y: float
    z: float

    @staticmethod
    def broadcast(x):
        """ Broadcast single component to entire vector """
        return Vec3f(x, x, x)

    @staticmethod
    def zero() -> Vec3f:
        """ Build zero vector """
        return Vec3f(0, 0, 0)
    
    @staticmethod
    def from_np(n: np.ndarray) -> Vec3f:
        """ Construct vec3f from numpy ndarray """
        return Vec3f(n[0], n[1], n[2])

    @staticmethod
    def _check_operand(o: Any):
        """ Validate vec3f operator right operand type """
        if not isinstance(o, Vec3f): raise TypeError("vec3f operator operand should be vec3f")

    def into_np(self) -> np.ndarray:
        """ Convert vec3f into numpy ndarray """
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    def __add__(l, r): #pyright: ignore
        Vec3f._check_operand(r)
        return Vec3f(l.x + r.x, l.y + r.y, l.z + r.z)

    def __sub__(l, r): #pyright: ignore
        Vec3f._check_operand(r)
        return Vec3f(l.x - r.x, l.y - r.y, l.z - r.z)

    def __mul__(l, r): #pyright: ignore
        Vec3f._check_operand(r)
        return Vec3f(l.x * r.x, l.y * r.y, l.z * r.z)

    def __truediv__(l, r): #pyright: ignore
        Vec3f._check_operand(r)
        return Vec3f(l.x / r.x, l.y / r.y, l.z / r.z)

    def __neg__(self):
        return Vec3f(-self.x, -self.y, -self.z)
    
    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def dot(l, r): #pyright: ignore
        """ Calculate vector dot product """
        return l.x * r.x + l.y * r.y + l.z * r.z

    def cross(l, r): #pyright: ignore
        """ Calculate vector cross product.  """
        return Vec3f(
                l.y * r.z - l.z * r.y,
                l.z * r.x - l.x * r.z,
                l.x * r.y - l.y * r.x
        )

    def length(self):
        """ Calculate vector euclidean length """
        return math.sqrt(self.dot(self))

    def normalized(self):
        """ Calculate unit vector with direction of self """
        return self / Vec3f.broadcast(self.length())

    def tobytes(self) -> bytes:
        """ Convert 3-component vector into byte array """
        return np.float32(np.array([self.x, self.y, self.z])).tobytes()
