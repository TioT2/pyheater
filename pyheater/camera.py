# Camera implementation module

import glfw

from mat4 import *
from vec3 import *

class Camera:
    """ Camera controller class """

    def __init__(self):
        self._location = Vec3f(50.0, 50.0, 50.0)
        self._direction = Vec3f.broadcast(-1.0).normalized()
        self._approx_up = Vec3f(0.0, 0.0, 1.0)

        self.movement_speed = 32.0
        self.rotation_speed = 4.0

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
        near = 0.1

        mwh = min(width, height)
        wp = near * width / mwh / 2
        hp = near * height / mwh / 2

        self._projection = Mat4f.proj_frustum_inf_far(-wp, wp, -hp, hp, near)

    @property
    def location(self) -> Vec3f:
        """ Get camera location """
        return self._location

    @location.setter 
    def location(self, val: Vec3f):
        """ Location setter """
        self._location = val
        self._valid_vp = False

    @property
    def direction(self) -> Vec3f:
        """ Get camera direction """
        return self._direction

    @direction.setter
    def direction(self, val: Vec3f):
        """ Set direction """
        self._direction = val.normalized()
        self._valid_vp = False

    def _ensure_vp(self):
        if not self._valid_vp:
            self._view = Mat4f.view(self._location, self._direction, self._approx_up)
            self._vp = self._projection * self._view
            self._valid_vp = True

    @property
    def view(self) -> Mat4f:
        self._ensure_vp()
        return self._view

    @property 
    def projection(self) -> Mat4f:
        self._ensure_vp()
        return self._projection

    @property
    def view_projection(self) -> Mat4f:
        """ Access view-projection camera matrix """
        self._ensure_vp()
        return self._vp

    def collect_glfw_input(self, window: Any) -> tuple[Vec3f, Vec3f]:
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

        movement = movement_axis * Vec3f.broadcast(dt * self.movement_speed)
        rotation = rotation_axis * Vec3f.broadcast(dt * self.rotation_speed)

        dir = self._direction.normalized()
        right = dir.cross(self._approx_up).normalized()
        up = right.cross(dir).normalized()

        self.set_location(self._location
            + dir   * Vec3f.broadcast(movement.x)
            + right * Vec3f.broadcast(movement.y)
            + up    * Vec3f.broadcast(movement.z)
        )

        azimuth = math.acos(dir.z)
        elevator = math.atan2(dir.y, dir.x) # math.copysign(1, dir.y) * math.acos(dir.x / math.sqrt(dir.x * dir.x + dir.y * dir.y))

        elevator -= rotation.x
        azimuth  += rotation.y

        azimuth = np.clip(azimuth, 0.01, np.pi - 0.01)

        self.set_direction(Vec3f(
            np.sin(azimuth) * np.cos(elevator),
            np.sin(azimuth) * np.sin(elevator),
            np.cos(azimuth)
        ))
