import numpy as np

from vec3 import Vec3f

class Mat4f:
    """ 4x4 floating-point matrix """

    def __init__(self, data=np.zeros((4, 4), dtype=np.float32)):
        """ Construct 4x4 matrix """
        self._data = np.array(data)

    @staticmethod
    def identity() -> Mat4f:
        """ Build identity matrix """
        
        return Mat4f(np.identity(4))

    @staticmethod
    def transform(v: Vec3) -> Mat4f:
        """ Build transform matrix """

        return Mat4f([
            [1.0, 0.0, 0.0, v.x],
            [0.0, 1.0, 0.0, v.y],
            [0.0, 0.0, 1.0, v.z],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def scale(v: Vec3) -> Mat4f:
        """ Build scale matrix """

        return Mat4f([
            [v.x, 0.0, 0.0, 0.0],
            [0.0, v.y, 0.0, 0.0],
            [0.0, 0.0, v.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def proj_frustum_inf_far(l: float, r: float, b: float, t: float, n: float) -> Mat4f:
        """ Build infinite-far frustum projection matrix """
        return Mat4f(np.array([
            [2.0 * n / (r - l),               0.0,  0.0,  0.0],
            [              0.0, 2.0 * n / (t - b),  0.0,  0.0],
            [(r + l) / (r - l), (t + b) / (t - b), -1.0, -1.0],
            [              0.0,               0.0,  0.0,  0.0],
        ]).T)

    @staticmethod
    def view(loc: Vec3f, dir: Vec3f, approx_up: Vec3f) -> Mat4f:
        """ Build view matrix """

        d = dir.normalized()
        r = d.cross(approx_up).normalized()
        u = r.cross(d).normalized()

        return Mat4f([
            [ r.x,  r.y,  r.z,  loc.dot(r)],
            [ u.x,  u.y,  u.z,  loc.dot(u)],
            [-d.x, -d.y, -d.z, -loc.dot(d)],
            [ 0.0,  0.0,  0.0,         1.0],
        ])


    def as_bytes(self) -> bytes:
        """ Convert matrix 4x4 matrix into column-major matrix byte array """
        return np.float32(self._data).T.tobytes()

    def __mul__(self, othr: Mat4f) -> Mat4f:
        """ Perform matrix multiplication """
        return Mat4f(self._data @ othr._data)

    def __getitem__(self, key: tuple[int, int]) -> float:
        return self._data[key]

    def __setitem__(self, key: tuple[int, int], value: float):
        self._data[key] = value

    def transform_vector(self, v: Vec3f) -> Vec3f:
        [x, y, z] = self._data[0:3, 0:3] @ np.array([v.x, v.y, v.z]).T
        return Vec3f(x, y, z)

    def transform_point(self, v: Vec3f) -> Vec3f:
        [x, y, z] = self._data[0:3, 0:4] @ np.array([v.x, v.y, v.z, 1.0]).T
        return Vec3f(x, y, z)

    def transform_4x4(self, v: Vec3f) -> Vec3f:
        [x, y, z, w] = self._data @ np.array([v.x, v.y, v.z, 1.0]).T
        return Vec3f(x / w, y / w, z / w)
