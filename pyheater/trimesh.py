import numpy as np
import importlib.resources as ilr
import wgpu
from itertools import product
from enum import Enum
from typing import Dict

from mat4 import *
from vec3 import *
from function_sample import FunctionSample

class Trimesh:
    """ OBJ model class """

    @staticmethod
    def parse_obj(source: str) -> Trimesh:
        """ Parse OBJ file text """

        vs = [[0.0, 0.0, 0.0]]
        ns = [[0.0, 0.0, 0.0]]
        ts = [[0.0, 0.0]]

        idx_map = {}
        vtx: list[list[float]] = []
        idx: list[int] = []
        
        for line in source.splitlines():
            line = line.strip().split()
            match line[0]:
                case 'v':
                    vs.append([float(line[1]), float(line[2]), float(line[3])])
                case 'vt':
                    ts.append([float(line[1]), float(line[2])])
                case 'vn':
                    ns.append([float(line[1]), float(line[2]), float(line[3])])
                case 'f':
                    def parsev(vert: str) -> int:
                        def parse(xis: str) -> int:
                            if len(xis) == 0: return 0
                            return int(xis)
                        [vi, ti, ni] = [parse(e.strip()) for e in vert.split('/')]
                        vnt = (vi, ni, ti)
                        if vnt not in idx_map:
                            idx_map[vnt] = len(vtx)
                            vt = vs[vi] + ns[ni] + ts[ti]
                            vtx.append(vt)
                        return idx_map[vnt]
                    
                    vbase = parsev(line[1])
                    vprev = parsev(line[2])

                    for vert in line[3:]:
                        vnext = parsev(vert)
                        idx.append(vbase)
                        idx.append(vprev)
                        idx.append(vnext)
                        vprev = vnext

        return Trimesh(np.array(vtx, dtype=np.float32).ravel(), np.array(idx, dtype=np.uint32))

    @staticmethod
    def isosurface(fs: FunctionSample, target: float) -> Trimesh:
        """ Build mesh for isosurface from `fs` at `target` value.  """

        (fw, fh, fd) = fs.shape
        fs_data = fs.read()

        # Triangle array
        idx = []
        vtx: list[Vec3f] = []
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

                i0 = get_ind(uv2c(x, y, z, -1, -1))
                i1 = get_ind(uv2c(x, y, z, -1,  0))
                i2 = get_ind(uv2c(x, y, z,  0,  0))
                i3 = get_ind(uv2c(x, y, z,  0, -1))

                idx.append(i0)
                idx.append(i1)
                idx.append(i2)

                idx.append(i0)
                idx.append(i2)
                idx.append(i3)

        plane(lambda x, y, z, d: fs_data[z + d, y, x], lambda x, y, z, u, v: (x + u, y + v, z + 0))
        plane(lambda x, y, z, d: fs_data[z, y + d, x], lambda x, y, z, u, v: (x + u, y + 0, z + v))
        plane(lambda x, y, z, d: fs_data[z, y, x + d], lambda x, y, z, u, v: (x + 0, y + u, z + v))

        nvtx = np.zeros((len(vtx) * 8,), dtype=np.float32)
        for i, vt in enumerate(vtx):
            nvtx[i * 8 : i * 8 + 3] = vt.into_np()

        return Trimesh(nvtx, np.array(idx, dtype=np.uint32))
    
    def get_polygon(self, ind: int) -> tuple[int, int, int]:
        """ Polygon indices """
        return (self.index[ind * 3 + 0], self.index[ind * 3 + 1], self.index[ind * 3 + 2])
    
    def get_vertex(self, ind: int) -> np.ndarray:
        return self.vertex[ind * 8 : (ind + 1) * 8]

    def build_vertex_normals(self):
        """ Recalculate vertex normals from positions """

        def add_normal(vn: np.ndarray, n: Vec3f):
            # if n.dot(Vec3f.from_np(vn)) < 0: n = -n
            vn += n.into_np()

        # Sum per-surface normals
        for ii in range(len(self.index) // 3):
            (i0, i1, i2) = self.get_polygon(ii)
            v0 = self.get_vertex(i0); v1 = self.get_vertex(i1); v2 = self.get_vertex(i2)

            n = Vec3f.cross(
                Vec3f.from_np(v1) - Vec3f.from_np(v0),
                Vec3f.from_np(v2) - Vec3f.from_np(v0)
            ).normalized()

            add_normal(v0[3:6], n)
            add_normal(v1[3:6], n)
            add_normal(v2[3:6], n)

        # Normalize them    
        for i in range(len(self.vertex) // 8):
            v = self.get_vertex(i)
            v[3:6] = Vec3f.from_np(v[3:6]).normalized().into_np()

    def __init__(self, vertex: np.ndarray, index: np.ndarray):
        """ Load OBJ model """

        self.vertex = vertex
        self.index = index

class TrimeshRenderCommon:
    """ Mesh model common functionality class """

    @staticmethod
    def build_uniform_data(world: Mat4f, vp: Mat4f, color: Vec3f) -> bytes:
        return (world * vp).tobytes() + world.inversed().tobytes() + color.tobytes() + bytes(np.float32(0))
    
    def build_pipeline(self, shader_module: wgpu.GPUShaderModule) -> wgpu.GPURenderPipeline:
        return self._device.create_render_pipeline(
            label = "Trimesh pipeline",
            layout = self._pipeline_layout,
            vertex = wgpu.VertexState(
                module = shader_module,
                buffers = [
                    wgpu.VertexBufferLayout(
                        array_stride = 32,
                        attributes = [
                            wgpu.VertexAttribute(
                                format = wgpu.VertexFormat.float32x3,
                                offset = 0,
                                shader_location = 0,
                            ),
                            wgpu.VertexAttribute(
                                format = wgpu.VertexFormat.float32x3,
                                offset = 12,
                                shader_location = 1,
                            ),
                        ]
                    )
                ]
            ),
            fragment = wgpu.FragmentState(
                module = shader_module,
                targets = [
                    wgpu.ColorTargetState(
                        format = wgpu.TextureFormat.bgra8unorm_srgb,
                    )
                ]
            ),
            primitive = wgpu.PrimitiveState(
                topology = "triangle-list",
            ),
            depth_stencil = wgpu.DepthStencilState(
                format = wgpu.TextureFormat.depth32float,
                depth_write_enabled = True,
                depth_compare = "greater-equal",
            ),
        )

    def __init__(self, device: wgpu.GPUDevice):
        self._device = device

        self._bgl = self._device.create_bind_group_layout(
            label = "Trimesh bg layout",
            entries = [
            wgpu.BindGroupLayoutEntry(
                binding = 0,
                visibility = wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX,
                buffer = wgpu.BufferBindingLayout()
            )
        ])

        self._pipeline_layout = self._device.create_pipeline_layout(
            label = "Trimesh pipeline layout",
            bind_group_layouts = [self._bgl]
        )

        self._pipelines: Dict[RenderTrimeshMode, wgpu.GPURenderPipeline] = {
            RenderTrimeshMode.default: self.build_pipeline(self._device.create_shader_module(
                label = "Trimesh default shader module",
                code = ilr.files("__main__").joinpath("trimesh.wgsl").read_text(encoding = "utf-8")
            )),
            RenderTrimeshMode.inplace_normals: self.build_pipeline(self._device.create_shader_module(
                label = "Trimesh inplace_normals shader module",
                code = ilr.files("__main__").joinpath("trimesh_inplace_normals.wgsl").read_text(encoding = "utf-8")
            ))
        }

class RenderTrimeshMode(Enum):
    default = "default"
    inplace_normals = "inplace_normals"

class RenderTrimesh:
    """ Mesh renderer """

    def __init__(self, device: wgpu.GPUDevice, model: Trimesh, mode: RenderTrimeshMode = RenderTrimeshMode.default):
        common = getattr(RenderTrimesh, '_common', None)
        if common is None:
            common = TrimeshRenderCommon(device)
            setattr(RenderTrimesh, '_common', common)

        # Public fields
        self.world = Mat4f.identity()
        self.color = Vec3f.broadcast(1.0)
        self.mode = mode

        env_data = common.build_uniform_data(Mat4f.identity(), Mat4f.identity(), Vec3f.zero())

        self._env_buffer = common._device.create_buffer(
            label = "Trimesh env buffer",
            usage = wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.UNIFORM,
            size = len(env_data),
        )

        self._bind_group = common._device.create_bind_group(
            label = "Model bind group",
            layout = common._bgl,
            entries = [
                wgpu.BindGroupEntry(
                    binding = 0,
                    resource = self._env_buffer,
                )
            ]
        )

        vt_bytes = model.vertex.tobytes()
        id_bytes = model.index.tobytes()

        self._vtx_count = len(model.vertex)
        self._idx_count = len(model.index)

        self._model_buffer = common._device.create_buffer(
            label = "RenderModel common buffer",
            usage = wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.INDEX,
            size = len(vt_bytes) + len(id_bytes),
        )

        common._device.queue.write_buffer(self._model_buffer, 0, vt_bytes)
        common._device.queue.write_buffer(self._model_buffer, len(vt_bytes), id_bytes)

    def render(self, enc: wgpu.GPURenderPassEncoder, view_projection: Mat4f):
        common = getattr(RenderTrimesh, '_common')
        common._device.queue.write_buffer(
            self._env_buffer,
            0,
            data = common.build_uniform_data(self.world, view_projection, self.color)
        )

        enc.set_pipeline(common._pipelines[self.mode])
        enc.set_bind_group(0, self._bind_group)

        enc.set_vertex_buffer(
            slot = 0,
            buffer = self._model_buffer,
            offset = 0,
            size = self._vtx_count * 4,
        )
        enc.set_index_buffer(
            buffer = self._model_buffer,
            offset = self._vtx_count * 4,
            index_format = wgpu.IndexFormat.uint32,
        )

        enc.draw_indexed(self._idx_count)