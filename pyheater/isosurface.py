# Isosurface renderer module
import wgpu
import importlib.resources as ilr
from typing import cast

from mat4 import *
from vec3 import *
from function_sample import *

class IsosurfaceCommon:
    """ Isosurface common rendering structures """

    @staticmethod
    def build_uniform(wvp: Mat4f, level: float, shape: tuple[int, int, int]) -> bytes:
        """ Build uniform buffer data """

        return wvp.tobytes() + np.float32(level).tobytes() + np.array(list(shape), dtype=np.int32).tobytes()

    def __init__(self, device: wgpu.GPUDevice):
        pass

class Isosurface:
    """ Generic function sample isosurface renderer """

    @staticmethod
    def _build_uniform(wvp: Mat4f, level: float, shape: tuple[int, int, int]) -> bytes:
        """ Build uniform buffer data """

        return wvp.tobytes() + np.float32(level).tobytes() + np.array(list(shape), dtype=np.int32).tobytes()

    def __init__(self, device: wgpu.GPUDevice):
        """ Constructor """

        self._device = device

        un_size = len(self._build_uniform(Mat4f.identity(), 0, (0, 0, 0)))

        self._env_buffer = self._device.create_buffer(
            label = "Isosurface env buffer",
            size = un_size,
            usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._vertex_buffer = None
        self._index_buffer = None

        bgl_entry_visibility = wgpu.ShaderStage.COMPUTE
        def bgl_entry(binding: int, ty: str, size: int) -> wgpu.BindGroupLayoutEntry:
            return wgpu.BindGroupLayoutEntry(
                binding = binding,
                visibility = bgl_entry_visibility,
                buffer = wgpu.BufferBindingLayout(
                    type = ty,
                    min_binding_size = size,
                )
            )
        
        bgl_entry_visibility = wgpu.ShaderStage.COMPUTE
        self._comp_bgl = self._device.create_bind_group_layout(
            label = "Isosurface compute bgl",
            entries = [
                bgl_entry(0, "uniform", un_size), # Env
                bgl_entry(1, "storage", 0), # Sample buffer
                bgl_entry(2, "storage", 0), # Vertex target buffer
            ]
        )

        bgl_entry_visibility = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self._render_bgl = self._device.create_bind_group_layout(
            label = "Isosurface render bgl",
            entries = [
                bgl_entry(0, "uniform", un_size), # Env
                bgl_entry(1, "read-only-storage", 0), # Vertex buffer
                bgl_entry(2, "read-only-storage", 0),
            ]
        )

        comp_sm = self._device.create_shader_module(
            label = "Isosurface compute shader",
            code = ilr.files("__main__").joinpath("isosurface_comp.wgsl").read_text(encoding = "utf-8")
        )
        render_sm = self._device.create_shader_module(
            label = "Isosurface render shader",
            code = ilr.files("__main__").joinpath("isosurface_render.wgsl").read_text(encoding = "utf-8")
        )

        self._comp_pipeline_layout = self._device.create_pipeline_layout(
            label = "Isosurface pipeline layout",
            bind_group_layouts = [self._comp_bgl],
        )
        self._comp_pipeline = self._device.create_compute_pipeline(
            label = "Isosurface compute pipeline",
            layout = self._comp_pipeline_layout,
            compute = wgpu.ProgrammableStage(module = comp_sm)
        )

        self._render_pipeline_layout = self._device.create_pipeline_layout(
            label = "Isosurface pipeline layout",
            bind_group_layouts = [self._render_bgl],
        )
        self._render_pipeline = self._device.create_render_pipeline(
            label = "Isosurface render pipeline",
            layout = self._render_pipeline_layout,
            vertex = wgpu.VertexState(
                module = render_sm,
                entry_point = "vs_main",
                buffers = []
            ),
            fragment = wgpu.FragmentState(
                module = render_sm,
                entry_point = "fs_main",
                targets = [
                    wgpu.ColorTargetState(
                        format = wgpu.TextureFormat.bgra8unorm_srgb
                    )
                ]
            ),
            depth_stencil = wgpu.DepthStencilState(
                format = wgpu.TextureFormat.depth32float,
                depth_compare = "greater-equal",
                depth_write_enabled = True,
            ),
            primitive = wgpu.PrimitiveState(
                topology = "triangle-list",
            ),
        )

    def set_sample(self, sample: FunctionSample | None):
        """ Set target function sample """

        prev_shape = (-1, -1, -1)
        if getattr(self, '_sample', None) is not None:
            prev_shape = cast(Any, self._sample).shape

        self._sample = sample
        if self._sample is None:
            return

        (w, h, d) = self._sample.shape

        # No need in buffer update
        if w * h * d <= prev_shape[0] * prev_shape[1] * prev_shape[2]:
            return

        self._vertex_buffer = self._device.create_buffer(
            label="Isosurface vertex buffer",
            size = w * h * d * 12,
            usage = wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE
        )

        self._comp_bg = self._device.create_bind_group(
            label = "Isosurface compute bind group",
            layout = self._comp_bgl,
            entries = [
                wgpu.BindGroupEntry(binding = 0, resource = self._env_buffer),
                wgpu.BindGroupEntry(binding = 1, resource = self._sample.buffer),
                wgpu.BindGroupEntry(binding = 2, resource = self._vertex_buffer),
            ]
        )

        self._render_bg = self._device.create_bind_group(
            label = "Isosurface render bind group",
            layout = self._render_bgl,
            entries = [
                wgpu.BindGroupEntry(binding = 0, resource = self._env_buffer),
                wgpu.BindGroupEntry(binding = 1, resource = self._vertex_buffer),
                wgpu.BindGroupEntry(binding = 2, resource = self._sample.buffer),
            ]
        )

    def compute(self, enc: wgpu.GPUComputePassEncoder, view_projection: Mat4f, level: float):
        """ Perform compute pass. """
        if self._sample is None: return

        # world matrix
        world = Mat4f.translate(self._sample.min) * Mat4f.scale(Vec3f(self._sample.step, self._sample.step, self._sample.step))

        # Run compute
        (w, h, d) = self._sample.shape

        # Update uniform buffer
        self._device.queue.write_buffer(self._env_buffer, 0, self._build_uniform(view_projection * world, level, (w, h, d)))

        enc.set_pipeline(self._comp_pipeline)
        enc.set_bind_group(0, self._comp_bg)
        enc.dispatch_workgroups(w - 1, h - 1, d - 1)

    def render(self, enc: wgpu.GPURenderPassEncoder):
        """ Perform rendering. """
        if self._sample is None: return

        (w, h, d) = self._sample.shape

        enc.set_pipeline(self._render_pipeline)
        enc.set_bind_group(0, self._render_bg)
        enc.draw(vertex_count = (w - 2) * (h - 2) * (d - 2) * 18)
