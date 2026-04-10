# Heat transfer simulation, PoC

from dataclasses import dataclass
import math
import sys
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

    def __add__(l, r):
        if not isinstance(r, Vec3f): raise TypeError("right vec3f operator operand should be vec3f")
        return Vec3f(l.x + r.x, l.y + r.y, l.z + r.z)

    def __sub__(l, r):
        if not isinstance(r, Vec3f): raise TypeError("right vec3f operator operand should be vec3f")
        return Vec3f(l.x - r.x, l.y - r.y, l.z - r.z)

    def __mul__(l, r):
        if not isinstance(r, Vec3f): raise TypeError("right vec3f operator operand should be vec3f")
        return Vec3f(l.x * r.x, l.y * r.y, l.z * r.z)

    def __truediv__(l, r):
        if not isinstance(r, Vec3f): raise TypeError("right vec3f operator operand should be vec3f")
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

    @staticmethod
    def sphere(r: f32) -> SDF:
        """ Build SDF of sphere with `r` radius.  """
        return SDF((lambda v: v.length() - r), (lambda v: v.normalized()))

    @staticmethod
    def box(d: Vec3f) -> SDF:
        """ Build box SDF with `d` dimensions. """
        def dist(v: Vec3f) -> float:
            q = Vec3f(abs(v.x) - d.x, abs(v.y) - d.y, abs(v.z) - d.z)
            return Vec3f(max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0)).length() + min(0.0, max(q.x, q.y, q.z))
        return SDF(dist)

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
            return self._grad(v) * Vec3f.broadcast(math.sign(self._dist(v)))

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

class FunctionSample:
    """ Sampled function value """

    def __init__(self, v0: Vec3f, v1: Vec3f, step: float):
        # Sampling step
        self._step = step

        # Bounding box
        self._min = Vec3f(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z))
        self._imax = Vec3f(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z))

        # Per-coordinate resolutions
        self._resx = int(math.ceil((self._imax.x - self._min.x) / step))
        self._resy = int(math.ceil((self._imax.y - self._min.y) / step))
        self._resz = int(math.ceil((self._imax.z - self._min.z) / step))

        # Calculate actual sampling range maximum from resolutions and step
        self._max = self._min + Vec3f(self._resx * step, self._resy * step, self._resz * step)

        # Fill contents with zeros
        self._data = np.zeros((self._resx, self._resy, self._resz))

    def get_cell_position(self, ix: int, iy: int, iz: int):
        """ Calculate world position of cell with (ix, iy, iz) coordindates.  """

        return self._min + Vec3f(ix, iy, iz) * Vec3f.broadcast(self._step)

    def sample_scalar_function(self, f):
        """ Fill sample with some function sampling result.  """

        sf = np.vectorize(lambda x, y, z: f(self.get_cell_position(x, y, z)))
        self._data = np.fromfunction(sf, self._data.shape, dtype=int)

    #def sample_vector_function(self, f):
    #    """ Sample some function that yields vec3f instead of float. Used as gradient field.  """
    #    def genr():
    #        for i in product(range(self._resx), range(self._resy), range(self._resz)):
    #            v = f(self.get_cell_position(*i))
    #            # piz... perfor... python!
    #            yield v.x
    #            yield v.y
    #            yield v.z
    #    self._data = np.fromiter(genr()).reshape((self._resz, self._resy, self._resx, 3))

    @property
    def min(self) -> Vec3f:
        """ Get minimum of function sampling coordinates """
        return self._min

    @property
    def max(self) -> Vec3f:
        """ Get actual maximum of function sampling coordinates """
        return self._max

    @property
    def imax(self) -> Vec3f:
        """ Get function sampling coordinate maximum from initial definition """
        return self._imax

    @property
    def step(self) -> float:
        """ Get per-coordinate sampling step """
        return self._step

    @property
    def shape(self) -> tuple[int, int, int]:
        """ Get (w, h, d) sample resolution tuple.  """
        return (self._resx, self._resy, self._resz)

    @property
    def data(self) -> np.array:
        """ Get sample contents organized in numpy array.  """
        return self._data

class Mesh:
    """ Indexed polygonal mesh. Negative index sign signs about polygon interrupt.  """

    def __init__(self, vtx: list[Vec3f] = [], idx: list[int] = []):
        """ Build mesh from vertex-index set.  """
        self._vtx = vtx
        self._idx = idx

    @property
    def vertices(self) -> list[Vec3f]:
        """ Get mesh vertex array """
        return self._vtx

    @property
    def indices(self) -> list[Vec3f]:
        """ Get mesh index array """
        return self._idx

    @property
    def polygons(self) -> list[list[Vec3f]]:
        """ Return list of polygons """
        ps = []
        p = []
        for i in self._idx:
            if i < 0:
                ps.append(p)
                p = []
            else:
                p.append(self._vtx[i])
        return ps

    def load_obj(self, file):
        """ Load obj file into current mesh. """
        raise NotImplementedError("TODO")

    def save_obj(self, file):
        """ Save current mesh to obj file. """
        for v in self._vtx:
            file.write(f"v {v.x} {v.y} {v.z}\n")
        file.write("f")
        for i in self._idx:
            if i < 0:
                file.write("\nf")
                continue
            file.write(f" {i + 1}")

def build_isosurface_mesh(fs: FunctionSample, target: float) -> Mesh:
    """ Build mesh for isosurface from `fs` at `target` value.  """

    (fw, fh, fd) = fs.shape

    # Triangle array
    idx = []
    vtx = []
    vt_map = {}

    def gen_point(r: tuple[int, int, int]) -> Vec3f:
        """ Generate actual point position by it's coordinate triple """

        x, y, z = r

        zeros = []
        # Encoded edge offsets
        edges = [0x01, 0x23, 0x45, 0x67, 0x04, 0x15, 0x26, 0x37, 0x02, 0x13, 0x46, 0x57]
        def decode(b): return b & 1, (b & 2) >> 1, (b & 4) >> 2

        for ecode in edges:
            dx0, dy0, dz0 = decode(ecode >> 4)
            dx1, dy1, dz1 = decode(ecode & 7)
            v0 = fs.data[z + dz0, y + dy0, x + dx0]
            v1 = fs.data[z + dz1, y + dy1, x + dx1]
            if (v0 >= target) == (v1 >= target):
                continue
            k = v0 / (v0 - v1)
            zeros.append(np.array([dx0 + k * (dx1 - dx0), dy0 + k * (dy1 - dy0), dz0 + k * (dz1 - dz0)]))
        c = zeros[0] if len(zeros) < 2 else np.mean(np.array(zeros), axis=0)
        cx, cy, cz = c
        return fs.min + Vec3f.broadcast(fs.step) * Vec3f(x + cx, y + cy, z + cz)

    def get_ind(r: tuple[int, int, int]) -> int:
        """ Get index of vertex by r coordinate triple """
        if r in vt_map: return vt_map[r]
        ind = len(vtx)
        vt_map[r] = ind
        vtx.append(gen_point(r))
        return ind

    def plane(ind_edge, uv2c):
        """ Traverse class of collinear edges and generate mesh for them. """

        for (z, y, x) in product(range(1, fd - 1), range(1, fh - 1), range(1, fw - 1)):

            # Check for sign change
            if (ind_edge(x, y, z, 0) >= target) == (ind_edge(x, y, z, 1) >= target):
                continue

            idx.append(get_ind(uv2c(x, y, z, -1, -1)))
            idx.append(get_ind(uv2c(x, y, z, -1,  0)))
            idx.append(get_ind(uv2c(x, y, z,  0,  0)))
            idx.append(get_ind(uv2c(x, y, z,  0, -1)))
            idx.append(-1)

    plane(lambda x, y, z, d: fs.data[z + d, y, x], lambda x, y, z, u, v: (x + u, y + v, z + 0))
    plane(lambda x, y, z, d: fs.data[z, y + d, x], lambda x, y, z, u, v: (x + u, y + 0, z + v))
    plane(lambda x, y, z, d: fs.data[z, y, x + d], lambda x, y, z, u, v: (x + 0, y + u, z + v))

    return Mesh(vtx, idx)

class HeatTransferSimulation:
    """ Simulation main class """

    def __init__(self, cond: FunctionSample):
        """ Constructor. Takes conductivity voxel map on input. """
        self._cond = cond
        self._heat = FunctionSample(cond.min, cond.imax, cond.step)

    @property
    def cond(self) -> FunctionSample:
        """ Get simulation conductivity map """
        return self._cond

    @property
    def heat(self) -> FunctionSample:
        """ Get simulation heat map """
        return self._heat

if __name__ == '__main__':
    # Build sphere shape
    shape = SDF.sphere(1.0).ring(0.08)\
        .substract(SDF.sphere(0.8).translate(Vec3f(1.0, 0.0, 0.0)))\
        .union(SDF.sphere(0.4).translate(Vec3f(0.0, 1.1, +0.6)))\
        .union(SDF.sphere(0.4).translate(Vec3f(0.0, 1.1, -0.6)))

    # Sample sphere SDF
    sample = FunctionSample(Vec3f(-1.5, -1.5, -1.5), Vec3f(1.5, 1.6, 1.5), 0.0333)
    sample.sample_scalar_function(lambda v: shape.dist(v))

    # Convert sphere into isosurface and emit it as OBJ file in stdout
    surface = build_isosurface_mesh(sample, 0.0)
    surface.save_obj(sys.stdout)
