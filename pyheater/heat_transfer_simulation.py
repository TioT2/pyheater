import numpy as np
from function_sample import FunctionSample
import wgpu
import importlib

__all__ = [
    'HeatTransferSimulation'
]

def build_env_data(step: float, env_temp: float, delta_time: float, shape: tuple[int, int, int]) -> bytes:
    """ Construct environment data bytes from parameters """

    float_d = np.array([step, env_temp, delta_time, 0], dtype=np.float32)
    int_d = np.array([shape[2], shape[1], shape[0], 0], dtype=np.int32)
    return np.concatenate((float_d, int_d.astype(np.float32))).tobytes()

class HeatTransferSimulation:
    """ GPU-based heat transfer simulation class """

    def __init__(self, cap: FunctionSample, cond: FunctionSample, device: wgpu.GPUDevice):
        """ Constructor. Takes capacity, conductivity and GPU device on input. """

        self._cap = cap
        self._cond = cond

        # Two buffers swapping during simulation
        self._temp = FunctionSample(cond.min, cond.imax, cond.step, device)
        self._temp_dst = FunctionSample(cond.min, cond.imax, cond.step, device)

        self._device = device

        shader_code = importlib.resources.files("__main__").joinpath("heat_transfer_simulation.wgsl").read_text(encoding="utf-8")
        self._shader = self._device.create_shader_module(code = shader_code)

        env_data_size = len(build_env_data(0, 0, 0, (0, 0, 0)))
        self._env = self._device.create_buffer(
            label = "Simulation environment uniform buffer",
            usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
            size = env_data_size,
        )

        def bgl_entry(binding: int, ty: str, min_size: int) -> wgpu.BindGroupLayoutEntry:
            return wgpu.BindGroupLayoutEntry(
                binding = binding,
                visibility = wgpu.ShaderStage.COMPUTE,
                buffer = wgpu.BufferBindingLayout(
                    type = ty,
                    min_binding_size = min_size,
                )
            )

        self._bind_group_layout = self._device.create_bind_group_layout(
            label = "Simulation bind group layout",
            entries = [
                bgl_entry(0, "uniform", env_data_size), # Env
                bgl_entry(1, "storage", 0), # Heat capacity
                bgl_entry(2, "storage", 0), # Conductivity
                bgl_entry(3, "storage", 0), # Heat source
                bgl_entry(4, "storage", 0), # Heat destination
            ]
        )
        self._pipeline_layout = self._device.create_pipeline_layout(
            label = "Simulation pipeline layout",
            bind_group_layouts = [self._bind_group_layout],
        )
        self._pipeline = self._device.create_compute_pipeline(
            label = "Simulation compute pipeline",
            layout = self._pipeline_layout,
            compute = wgpu.ProgrammableStage(
                entry_point = "main",
                module = self._shader,
            )
        )


        def create_bind_group(src, dst) -> wgpu.GPUBindGroup:
            def entry(i, b):
                return wgpu.BindGroupEntry(
                    binding = i,
                    resource = wgpu.BufferBinding(buffer = b)
                )
            return self._device.create_bind_group(
                label = "Heat bind group",
                layout = self._bind_group_layout,
                entries = [
                    entry(0, self._env),
                    entry(1, self._cap.buffer),
                    entry(2, self._cond.buffer),
                    entry(3, src),
                    entry(4, dst),
                ]
            )

        self._bind_group     = create_bind_group(self._temp.buffer, self._temp_dst.buffer)
        self._bind_group_dst = create_bind_group(self._temp_dst.buffer, self._temp.buffer)

    @property
    def capacity(self) -> FunctionSample:
        """ Get simulation heat capacity map """
        return self._cap

    @property
    def cond(self) -> FunctionSample:
        """ Get simulation conductivity map """
        return self._cond

    @property
    def temp(self) -> FunctionSample:
        """ Get simulation current temperature map """
        return self._temp

    def clear_temp_to(self, temp: float):
        """ Clear simulation temperature buffer to some constant value """
        self._temp.sample(lambda _: temp)

    def update(self, env_temp: float, delta_time: float):
        """ Perform next step of simulation """

        # Assert delta time correctness
        if delta_time < 0: raise Exception("Negative step time")

        # Update environment buffer
        self._device.queue.write_buffer(
            self._env,
            0,
            build_env_data(
                self._cond.step,
                env_temp,
                delta_time,
                self._cond.shape,
            )
        )

        # Clear dst buffer
        encoder = self._device.create_command_encoder(label = "Simulation update command encoder")

        # Get global shape
        (sx, sy, sz) = self._cond.shape

        # Copy source buffer to destination
        encoder.copy_buffer_to_buffer(self._temp.buffer, 0, self._temp_dst.buffer, 0, 4 * sx * sy * sz)

        # Encode compute pass
        pass_encoder = encoder.begin_compute_pass(label = "Compute pass encoder")
        pass_encoder.set_bind_group(index = 0, bind_group = self._bind_group)
        pass_encoder.set_pipeline(self._pipeline)
        pass_encoder.dispatch_workgroups(sx - 1, sy - 1, sz - 1)
        pass_encoder.end()

        # Submit compute
        self._device.queue.submit([encoder.finish()])

        # Swap bind groups and heat function samples
        self._bind_group, self._bind_group_dst = self._bind_group_dst, self._bind_group
        self._temp, self._temp_dst = self._temp_dst, self._temp
