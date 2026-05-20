# Heat transfer simulation main file

from dataclasses import dataclass
import numpy as np
import wgpu
import wgpu.utils.imgui as imgui_wgpu
import imgui_bundle.imgui as imgui
from rendercanvas.auto import RenderCanvas, loop
import rendercanvas
from typing import Any, cast

from vec3 import Vec3f
from time_controller import Timer
from function_sample import FunctionSample
from sdf import SDF
from heat_transfer_simulation import HeatTransferSimulation
from isosurface import Isosurface
from trimesh import Trimesh, RenderTrimesh, RenderTrimeshMode
from camera import Camera
from enum import Enum

class SurfaceState(Enum):
    Isoterm = "isoterm"
    Surface = "surface"
surface_states = [SurfaceState.Isoterm, SurfaceState.Surface]

class RenderState:
    """ GUI management """

    def __init__(self):
        """ Initialize render state """

        self.enable_rei = False
        self.isoterm_level = 274.0
        self.run_simulation = True
        self.step_time_coef = 1.0
        self.env_temp = 273.0
        self.env_heat_xchg = 0.0

        self.cam_rotation_speed = 2.5
        self.cam_movement_speed = 40.0

        self.surf_state = 0
        self.surf_state_changed = False

        self._show_camera = False
        self._show_misc = False

    def gui(self):
        """ ImGUI gui rendering function """

        # imgui.set_next_window_pos((10, 10), cond=imgui.Cond_.once)
        imgui.begin("Simulation", flags=imgui.WindowFlags_.always_auto_resize)

        self.env_temp = imgui.slider_float("Environment temperature", self.env_temp, 0, 1000)[1]
        self.env_heat_xchg = imgui.slider_float("Environment heat exchange", self.env_heat_xchg, 0, 1000000)[1]
        self.step_time_coef = imgui.slider_float("Step time coefficent", self.step_time_coef, 0.5, 5.0)[1]
        self.isoterm_level = imgui.slider_float("Isoterm level", self.isoterm_level, 0, 1000)[1]
        self.run_simulation = imgui.checkbox("Run simulation", self.run_simulation)[1]

        self.surf_state_changed, self.surf_state = imgui.list_box(
            "Surface rendering mode",
            self.surf_state,
            [a.value for a in surface_states]
        )

        self._show_misc = imgui.checkbox("Show misc", self._show_misc)[1]
        if self._show_misc:
            self._show_camera = imgui.checkbox("Show camera", self._show_camera)[1]
            if self._show_camera:
                self.cam_rotation_speed = imgui.slider_float("Rotation speed", self.cam_rotation_speed, 0.0, 10.0)[1]
                self.cam_movement_speed = imgui.slider_float("Movement speed", self.cam_movement_speed, 0.0, 200.0)[1]

            self.enable_rei = imgui.checkbox("Enable rei", self.enable_rei)[1]

        imgui.end()

class Render:
    """ Class that manages simulation presentation """

    def __init__(
            self,
            context: rendercanvas.contexts.WgpuContext,
            window: Any,
            imgui_renderer: imgui_wgpu.ImguiRenderer,
            device: wgpu.GPUDevice,
            adapter: wgpu.GPUAdapter,
            simulation: HeatTransferSimulation,
            surface_mesh: Trimesh,
        ):
        """ Initialize renderer """
        self._timer = Timer()

        self._imgui_renderer = imgui_renderer

        self._context = context
        self._window = window
        self._device = device
        self._adapter = adapter

        self._state = RenderState()

        context.configure(device=device, format=context.get_preferred_format(adapter))

        self._depth_target = None

        self._simulation = simulation
        self._camera = Camera()

        self._rei = RenderTrimesh(
            self._device,
            Trimesh.parse_obj(open('.local/rei.obj').read())
        )
        self._surface_mesh = RenderTrimesh(self._device, surface_mesh, RenderTrimeshMode.inplace_normals)

        self._isoterm = Isosurface(self._device)
        self._isoterm.set_sample(self._simulation.temp)

        self._state = RenderState()
        self._imgui_renderer.set_gui(lambda: self._state.gui())

    def _depth_buffer(self, width: int, height: int) -> wgpu.GPUTexture:
        """ Return depth buffer of requested resolution """

        if self._depth_target is None or self._depth_target.width != width or self._depth_target.height != height:
            self._depth_target = self._device.create_texture(
                label = "Render depth texture",
                size = (width, height, 1),
                format = wgpu.TextureFormat.depth32float,
                usage = wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
        return self._depth_target

    def update(self):
        """ Perform next step """

        # Update timer and simulation
        self._timer.update()

        if self._state.run_simulation:
            self._simulation.env_temp = self._state.env_temp
            self._simulation.env_heat_xchg = self._state.env_heat_xchg
            self._simulation.update(self._timer.delta_time * self._state.step_time_coef)

        # Control camera
        self._camera.rotation_speed = self._state.cam_rotation_speed
        self._camera.movement_speed = self._state.cam_movement_speed

        amov, arot = self._camera.collect_glfw_input(self._window)
        self._camera.control_flycam(self._timer.delta_time, amov, arot)

        # Render (nothing for now)
        surf_texture = cast(wgpu.GPUTexture, self._context.get_current_texture())

        self._camera.set_screen_resolution(surf_texture.width, surf_texture.height)

        command_encoder = self._device.create_command_encoder()

        comp_pass = command_encoder.begin_compute_pass()
        self._isoterm.compute(comp_pass, self._camera.view_projection, self._state.isoterm_level)
        comp_pass.end()

        render_pass = command_encoder.begin_render_pass(
            color_attachments = [
                wgpu.RenderPassColorAttachment(
                    view = surf_texture.create_view(),
                    clear_value = (0.30, 0.47, 0.80, 1.00),
                    load_op = wgpu.LoadOp.clear,
                    store_op = wgpu.StoreOp.store,
                )
            ],
            depth_stencil_attachment = wgpu.RenderPassDepthStencilAttachment(
                view = self._depth_buffer(surf_texture.width, surf_texture.height).create_view(),
                depth_clear_value = 0.0,
                depth_load_op = wgpu.LoadOp.clear,
                depth_store_op = wgpu.StoreOp.store,
            )
        )
        if self._state.enable_rei: self._rei.render(render_pass, self._camera.view_projection)

        match surface_states[self._state.surf_state]:
            case SurfaceState.Isoterm:
                self._isoterm.render(render_pass)
            case SurfaceState.Surface:
                self._surface_mesh.render(render_pass, self._camera.view_projection)

        render_pass.end()

        self._device.queue.submit([command_encoder.finish()])

        # Display ImGUI
        self._imgui_renderer.render()
    
if __name__ == '__main__':
    # Perform rendering
    canvas = RenderCanvas(
        size = (640, 480),
        title = "WGPU rendering example with $backend",
        update_mode = "continuous",
        max_fps = 60,
        vsync = True
    )
    window = cast(Any, canvas)._window # uuuh shitcode
    context = canvas.get_wgpu_context()

    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync()

    # Build sphere shape
    shape = SDF.sphere(10.0).ring(0.8)\
        .substract(SDF.sphere(8.0).translate(Vec3f(10.0, 0.0, 0.0)))\
        .union(SDF.sphere(4.0).translate(Vec3f(0.0, 11.0, +6.0)))\
        .union(SDF.sphere(4.0).translate(Vec3f(0.0, 11.0, -6.0)))

    # Sample sphere SDF
    # sample_cell_size = 0.0333
    sample_cell_size = 0.5
    sample = FunctionSample(Vec3f(-20.0, -20.0, -20.0), Vec3f(20.0, 20.0, 20.0), sample_cell_size, device)
    sample.sample(lambda v: shape.dist(v))

    sample_data = sample.read()

    surface_mesh = Trimesh.isosurface(sample, 0)

    capacity = FunctionSample(sample.min, sample.imax, sample.step, device)
    capacity.write(np.sign(sample_data) * -460)

    conductivity = FunctionSample(sample.min, sample.imax, sample.step, device)
    conductivity.write(np.sign(sample_data) * -80)

    simulation = HeatTransferSimulation(sample, 460, 80, device)
    simulation.clear_temp_to(22 + 271)

    imgui_renderer = imgui_wgpu.ImguiRenderer(device, canvas)

    render = Render(context, window, imgui_renderer, device, adapter, simulation, surface_mesh)

    # Start rendering
    canvas.request_draw(lambda: render.update())
    loop.run()
