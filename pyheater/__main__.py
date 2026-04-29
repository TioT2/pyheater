# Heat transfer simulation main file

from dataclasses import dataclass
import math
import sys
import numpy as np
import wgpu
import glfw
from rendercanvas.auto import RenderCanvas, loop

from vec3 import Vec3f
from mat4 import Mat4f
from timer import Timer
from function_sample import FunctionSample
from sdf import SDF
from heat_transfer_simulation import HeatTransferSimulation
import mesh

class Camera:
    """ Camera controller class """

    def __init__(self):
        self._location = Vec3f(10.0, 10.0, 10.0)
        self._direction = Vec3f.broadcast(-1.0).normalized()
        self._approx_up = Vec3f(0.0, 0.0, 1.0)

        self._valid_vp = False
        self._view = Mat4f.identity()
        self._projection = Mat4f.identity()
        self._vp = Mat4f.identity()

        self.set_screen_resolution(800, 600)

    def set_location(self, loc: Vec3f):
        """ Set camera location """
        self._valid_vp = False
        self._location = loc

    def set_direction(self, dir: Vec3f):
        """ Set camera direction """
        self._valid_vp = False
        self._direction = dir

    def set_screen_resolution(self, width: float, height: float):
        """ Update camera to match screen resolution """

        self._valid_vp = False
        mwh = min(width, height)
        wp = width / mwh / 2
        hp = height / mwh / 2

        self._projection = Mat4f.proj_frustum_inf_far(-wp, wp, -hp, hp, 1.0)

    @property
    def location(self):
        """ Get camera location """
        return self._location

    @property
    def direction(self):
        """ Get camera direction """
        return self._direction

    @property
    def view_projection(self):
        """ Access view-projection camera matrix """

        if not self._valid_vp:
            self._view = Mat4f.view(self._location, self._direction, self._approx_up)
            self._vp = self._projection * self._view
            self._valid_vp = True
        return self._vp

    def collect_glfw_input(self, window: glfw.Window) -> tuple[Vec3f, Vec3f]:
        """ Collect (move, rotation) input tuple from GLFW window """
        def axis(p, n):
            return float(glfw.get_key(window, p) == glfw.PRESS) - float(glfw.get_key(window, n) == glfw.PRESS)

        move = Vec3f(
            axis(glfw.KEY_W, glfw.KEY_S),
            axis(glfw.KEY_D, glfw.KEY_A),
            axis(glfw.KEY_R, glfw.KEY_F),
        )
        rotate = Vec3f(
            axis(glfw.KEY_RIGHT, glfw.KEY_LEFT),
            axis(glfw.KEY_DOWN, glfw.KEY_UP),
            0
        )

        return move, rotate

    def control_flycam(self, dt: float, movement_axis: Vec3f, rotation_axis: Vec3f):
        """ Perform camera control for given delta time, movement axis and camera rotation axis XY """

        movement = movement_axis * Vec3f.broadcast(dt * 256)
        rotation = rotation_axis * Vec3f.broadcast(dt * 256)

        dir = self._direction
        right = dir.cross(self._approx_up).normalized()
        up = right.cross(dir).normalized()

        self.set_location(Vec3f.zero()
            + dir   * Vec3f.broadcast(movement.x)
            + right * Vec3f.broadcast(movement.y)
            + up    * Vec3f.broadcast(movement.z)
        )
        azimuth = math.acos(dir.z)
        elevator = math.copysign(1, dir.y) * math.acos(dir.x / math.sqrt(dir.x * dir.x + dir.y * dir.y))

        elevator -= rotation.x
        azimuth  += rotation.y

        azimuth = np.clip(azimuth, 0.01, np.pi - 0.01)

        self.set_direction(Vec3f(
            np.sin(azimuth) * np.cos(elevator),
            np.sin(azimuth) * np.sin(elevator),
            np.cos(azimuth)
        ))

class Render:
    """ Structure that displays game contents """

    def __init__(self, context, window: glfw.Window, device: wgpu.GPUDevice, adapter: wgpu.GPUAdapter, simulation: HeatTransferSimulation):
        """ Initialize renderer """
        self._timer = Timer()

        self._context = context
        self._window = window
        self._device = device
        self._adapter = adapter

        context.configure(device=device, format=context.get_preferred_format(adapter))

        self._isoterm_level = None
        self._env_temp = 22 + 271
        self._simulation = simulation
        self._camera = Camera()

    def set_env_temp(self, temp: float):
        """ Set environment temperature (K) """

        if temp < 0: raise Exception('Negative temperatures does not exist')

        self._env_temp = temp

    def set_isoterm_level(self, isoterm_level: float | None):
        """ Set rendered isoterm level """
        self._isoterm_level = isoterm_level

    def update(self):
        """ Render next step """

        # Update timer and simulation
        self._timer.update()
        self._simulation.update(self._env_temp, self._timer.delta_time)

        # Control camera?
        amov, arot = self._camera.collect_glfw_input(self._window)
        self._camera.control_flycam(self._timer.delta_time, amov, arot)

        # Render
        surf_texture = self._context.get_current_texture()
    
if __name__ == '__main__':
    # Perform rendering
    canvas = RenderCanvas(
        size = (640, 480),
        title = "WGPU rendering example with $backend",
        update_mode = "continuous",
        max_fps = 60,
        vsync = True
    )
    window = canvas._window # uuuh shitcode
    context = canvas.get_wgpu_context()

    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync()

    # Build sphere shape
    shape = SDF.sphere(1.0).ring(0.08)\
        .substract(SDF.sphere(0.8).translate(Vec3f(1.0, 0.0, 0.0)))\
        .union(SDF.sphere(0.4).translate(Vec3f(0.0, 1.1, +0.6)))\
        .union(SDF.sphere(0.4).translate(Vec3f(0.0, 1.1, -0.6)))

    # Sample sphere SDF
    sample_cell_size = 0.0333
    sample = FunctionSample(Vec3f(-1.5, -1.5, -1.5), Vec3f(1.5, 1.6, 1.5), sample_cell_size, device)
    sample.sample(lambda v: shape.dist(v))

    sample_data = sample.read()

    capacity = FunctionSample(sample.min, sample.imax, sample.step, device)
    capacity.write(np.abs(sample_data) * -460)

    conductivity = FunctionSample(sample.min, sample.imax, sample.step, device)
    conductivity.write(np.abs(sample_data) * -80)

    simulation = HeatTransferSimulation(capacity, conductivity, device)
    simulation.clear_temp_to(22 + 271)

    render = Render(context, window, device, adapter, simulation)

    # Start rendering
    canvas.request_draw(lambda: render.update())
    loop.run()
