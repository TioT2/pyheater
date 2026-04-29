from itertools import product
import numpy as np

from vec3 import Vec3f
from function_sample import FunctionSample

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
    fs_data = fs.read()

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
            v0 = fs_data[z + dz0, y + dy0, x + dx0]
            v1 = fs_data[z + dz1, y + dy1, x + dx1]
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

    plane(lambda x, y, z, d: fs_data[z + d, y, x], lambda x, y, z, u, v: (x + u, y + v, z + 0))
    plane(lambda x, y, z, d: fs_data[z, y + d, x], lambda x, y, z, u, v: (x + u, y + 0, z + v))
    plane(lambda x, y, z, d: fs_data[z, y, x + d], lambda x, y, z, u, v: (x + 0, y + u, z + v))

    return Mesh(vtx, idx)
