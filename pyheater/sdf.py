import math

from vec3 import Vec3f

class SDF:
    """ Signed Distance Function class. Used for surface construction.  """

    def __init__(self, dist, grad = None):
        """ SDF constructor. Takes SDF (Vec3f -> Float function) and (optionally) SDF gradient function (Vec3f -> Vec3f).
        If gradient is None, it's calculated by manually from small SDF variations.  """
        self._dist = dist
        if grad == None:
            def _def_grad(v: Vec3f) -> Vec3f:
                """ Default by-definition gradient function. Much more slower and imprecise in comparison with manual gradient, calculates SDF six times.  """
                dxi = 0.001
                dfdx = (dist(v + Vec3f(+dxi,  0.0,  0.0)) - dist(v - Vec3f(-dxi,  0.0,  0.0))) / (2.0 * dxi)
                dfdy = (dist(v + Vec3f( 0.0, +dxi,  0.0)) - dist(v - Vec3f( 0.0, -dxi,  0.0))) / (2.0 * dxi)
                dfdz = (dist(v + Vec3f( 0.0,  0.0, +dxi)) - dist(v - Vec3f( 0.0,  0.0, -dxi))) / (2.0 * dxi)
                return Vec3f(dfdx, dfdy, dfdz)
            self._grad = _def_grad
        else:
            self._grad = grad

    def dist(self, at: Vec3f) -> float:
        """ Calculate SDF value at point.  """
        return self._dist(at)

    def grad(self, at: Vec3f) -> Vec3f:
        """ Calculate SDF gradient at point.  """

        return self._grad(at)

    def translate(self, dv: Vec3f) -> SDF:
        """ Offset SDF by `dv` vector.  """

        return SDF(lambda v: self._dist(v - dv), lambda v: self._grad(v - dv))

    def inflate(self, r: float) -> SDF:
        """ Inflate SDF by some distance.  """

        if r < 0: raise ValueError("SDF cannot be inflated by negative value")

        return SDF(lambda v: self._dist(v) + r, self._grad)

    def ring(self, r: float) -> SDF:
        """ Transform SDF to the 'ring'.  """

        if r < 0: raise ValueError("SDF ring radius should not be negative")

        def dist(v: Vec3f) -> float:
            return abs(self._dist(v)) - r
        def grad(v: Vec3f) -> Vec3f:
            return self._grad(v) * Vec3f.broadcast(math.copysign(1.0, self._dist(v)))

        return SDF(dist, grad)

    def inverse(self) -> SDF:
        """ Reverse SDF.  """

        return SDF(lambda v: -self._dist(v), lambda v: -self._grad(v))

    def union(self, other: SDF) -> SDF:
        """ Calculate rough object union. Calculated through SDF minimum.  """

        def dist(v: Vec3f) -> float:
            return min(self._dist(v), other._dist(v))
        def grad(v: Vec3f) -> Vec3f:
            return self._grad(v) if self._dist(v) < other._dist(v) else other._grad(v)
        return SDF(dist, grad)

    def intersect(self, other: SDF) -> SDF:
        """ Calculate object intersection.  """

        def dist(v: Vec3f) -> float:
            return max(self._dist(v), other._dist(v))
        def grad(v: Vec3f) -> Vec3f:
            return self._grad(v) if self._dist(v) > other._dist(v) else other._dist(v)
        return SDF(dist, grad)

    def substract(self, other: SDF) -> SDF:
        """ Substract one SDF from another. Equivalent to self.intersection(other.inverse()).  """

        def dist(v: Vec3f) -> float:
            return max(self._dist(v), -other._dist(v))
        def grad(v: Vec3f) -> Vec3f:
            return self._grad(v) if self._dist(v) > -other._dist(v) else -other._grad(v)
        return SDF(dist, grad)

    @staticmethod
    def sphere(r: float) -> SDF:
        """ Build SDF of sphere with `r` radius.  """
        return SDF((lambda v: v.length() - r), (lambda v: v.normalized()))

    @staticmethod
    def box(d: Vec3f) -> SDF:
        """ Build box SDF with `d` dimensions. """
        def dist(v: Vec3f) -> float:
            q = Vec3f(abs(v.x) - d.x, abs(v.y) - d.y, abs(v.z) - d.z)
            return Vec3f(max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0)).length() + min(0.0, max(q.x, q.y, q.z))
        return SDF(dist)
