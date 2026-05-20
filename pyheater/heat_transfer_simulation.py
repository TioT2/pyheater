import numpy as np
from function_sample import FunctionSample
import wgpu
import importlib.resources as ilr

__all__ = [
    'HeatTransferSimulation'
]

def build_env_data(
        step: float,
        conductivity: float,
        capacity: float,
        env_heat_xchg: float,
        env_temp: float,
        delta_time: float,
        shape: tuple[int, int, int]
    ) -> bytes:
    """ Construct environment data bytes from parameters """
    fs = np.array([
        step, env_temp, delta_time, conductivity,
        capacity, env_heat_xchg, 0, 0,
    ], dtype = np.float32)
    return fs.tobytes() + np.array([shape[2], shape[1], shape[0], 0], dtype=np.int32).tobytes()

class HeatTransferSimulation:
    """ GPU-based heat transfer simulation class """

    def __init__(self, solid: FunctionSample, cap: float, cond: float, device: wgpu.GPUDevice):
        """ Constructor. Takes capacity, conductivity and GPU device on input. """

        self._device = device
        self._capacity = cap
        self._conductivity = cond

        cell_buffer = np.packbits((solid.read() < 0).ravel(), axis=0, bitorder="little")
        cell_buffer.resize(((len(cell_buffer) + 3) // 4 * 4,))

        # Constant cell bitflag buffer
        self._cell_buffer = self._device.create_buffer_with_data(
            label = "Simulation cell flag buffer",
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            data = cell_buffer.tobytes(),
        )

        # Zero
        self._env_temp = 271.0
        self._env_heat_xchg = 0.0

        # Two buffers swapping during simulation
        self._temp = FunctionSample(solid.min, solid.imax, solid.step, device)
        self._temp_dst = FunctionSample(solid.min, solid.imax, solid.step, device)

        self._device = device

        # Read shader code from source directory
        shader_code = ilr.files("__main__").joinpath("heat_transfer_simulation.wgsl").read_text(encoding="utf-8")

        # Compile shader
        self._shader = self._device.create_shader_module(code = shader_code)

        env_data_size = len(build_env_data(0, 0, 0, 0, 0, 0, (0, 0, 0)))
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
                bgl_entry(1, "read-only-storage", 0), # Cell buffer
                bgl_entry(2, "storage", 0), # Heat source
                bgl_entry(3, "storage", 0), # Heat destination
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

        def bg_entry(i: int, b: wgpu.GPUBuffer):
            return wgpu.BindGroupEntry(
                binding = i,
                resource = wgpu.BufferBinding(buffer = b)
            )

        self._bind_group = self._device.create_bind_group(
            label = "Heat bind group",
            layout = self._bind_group_layout,
            entries = [
                bg_entry(0, self._env),
                bg_entry(1, self._cell_buffer),
                bg_entry(2, self._temp.buffer),
                bg_entry(3, self._temp_dst.buffer),
            ]
        )

    @property
    def capacity(self) -> float:
        """ Get simulation heat capacity """
        return self._capacity

    @property
    def cond(self) -> float:
        """ Get simulation conductivity """
        return self._conductivity

    @property
    def temp(self) -> FunctionSample:
        """ Get simulation current temperature map """
        return self._temp

    def clear_temp_to(self, temp: float):
        """ Set object constant temperature """
        self._temp_dst.sample(lambda _: temp)

    @property
    def env_temp(self) -> float:
        """ Get environment absolute temperature """
        return self._env_temp

    @env_temp.setter 
    def env_temp(self, temp: float):
        """ Set environment aboslute temperature """
        if temp < 0: raise ValueError("Absolute temperature cannot be negative")
        self._env_temp = temp

    @property
    def env_heat_xchg(self) -> float:
        """ Get environment heat exchange coefficent """
        return self._env_heat_xchg

    @env_heat_xchg.setter
    def env_heat_xchg(self, val: float):
        """ Set environment heat exchange coefficent """
        if val < 0: raise ValueError("Heat exchange coefficent cannot be negative")
        self._env_heat_xchg = val

    def update(self, delta_time: float):
        """ Perform next step of simulation """

        # Assert delta time correctness
        if delta_time < 0: raise Exception("Negative step time")

        # Update environment buffer
        self._device.queue.write_buffer(
            self._env,
            0,
            build_env_data(
                self._temp.step,
                self._conductivity,
                self._capacity,
                self._env_heat_xchg,
                self._env_temp,
                delta_time,
                self._temp.shape,
            )
        )

        # Clear dst buffer
        encoder = self._device.create_command_encoder(label = "Simulation update command encoder")

        # Get global shape
        (sx, sy, sz) = self._temp.shape

        # Update current simulation temperature
        encoder.copy_buffer_to_buffer(self._temp_dst.buffer, 0, self._temp.buffer, 0, 4 * sx * sy * sz)

        # Encode compute pass
        pass_encoder = encoder.begin_compute_pass(label = "Compute pass encoder")
        pass_encoder.set_bind_group(index = 0, bind_group = self._bind_group)
        pass_encoder.set_pipeline(self._pipeline)
        pass_encoder.dispatch_workgroups(sx, sy, sz)
        pass_encoder.end()

        # Submit compute
        self._device.queue.submit([encoder.finish()])
