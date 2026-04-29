from dataclasses import dataclass
import math

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
    def zero():
        """ Build zero vector """
        return Vec3f(0, 0, 0)

    def _check_operand(o):
        """ Validate vec3f operator right operand type """
        if not isinstance(o, Vec3f): raise TypeError("vec3f operator operand should be vec3f")
    
    def __add__(l, r):
        Vec3f._check_operand(r)
        return Vec3f(l.x + r.x, l.y + r.y, l.z + r.z)

    def __sub__(l, r):
        Vec3f._check_operand(r)
        return Vec3f(l.x - r.x, l.y - r.y, l.z - r.z)

    def __mul__(l, r):
        Vec3f._check_operand(r)
        return Vec3f(l.x * r.x, l.y * r.y, l.z * r.z)

    def __truediv__(l, r):
        Vec3f._check_operand(r)
        return Vec3f(l.x / r.x, l.y / r.y, l.z / r.z)

    def __neg__(self):
        return Vec3f(-self.x, -self.y, -self.z)

    def dot(l, r):
        """ Calculate vector dot product """
        return l.x * r.x + l.y * r.y + l.z * r.z

    def cross(l, r):
        """ Calculate vector cross product.  """
        return Vec3f(
                l.y * r.z - l.z * r.y,
                l.x * r.z - l.z * r.x,
                l.x * r.y - l.y * r.x
        )

    def length(self):
        """ Calculate vector euclidean length """
        return math.sqrt(self.dot(self))

    def normalized(self):
        """ Calculate unit vector with direction of self """
        return self / Vec3f.broadcast(self.length())
