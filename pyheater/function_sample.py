import math
import numpy as np
import wgpu

from vec3 import Vec3f

class FunctionSample:
    """ GPU-located sampled function value """

    def __init__(self, v0: Vec3f, v1: Vec3f, step: float, device: WGPU.GPUDevice):
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

        # Save device and create target buffer
        self._device = device
        self._buffer = device.create_buffer(
            size = 4 * self._resx * self._resy * self._resz,
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            mapped_at_creation = False,
        )

    def get_cell_position(self, ix: int, iy: int, iz: int):
        """ Calculate world position of cell with (ix, iy, iz) coordindates.  """

        return self._min + Vec3f(ix, iy, iz) * Vec3f.broadcast(self._step)

    def sample(self, f):
        """ Sample scalar function. """
        sf = np.vectorize(lambda x, y, z: f(self.get_cell_position(x, y, z)), otypes=[np.float32])
        self.write(np.fromfunction(sf, (self._resz, self._resy, self._resx), dtype=int))

    def read(self) -> np.array:
        """ Read sample data from GPU """
        mv = self._device.queue.read_buffer(self._buffer)
        return np.frombuffer(mv, dtype=np.float32).reshape((self._resz, self._resy, self._resx))

    def write(self, data: np.array):
        """ Write sample data to GPU """
        if data.shape != (self._resz, self._resy, self._resx):
            raise Exception('Invalid data dimensions')
        self._device.queue.write_buffer(self._buffer, 0, data.tobytes())

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
    def buffer(self) -> wgpu.GPUBuffer:
        """ Return underlying WGPU buffer """
        return self._buffer
