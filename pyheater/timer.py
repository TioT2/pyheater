import time

class Timer:
    """ Time controller class """

    def __init__(self):
        """ Initialize timer. Starts from now. """

        now = time.perf_counter()

        self._start = now
        self._now = now
        self._delta_time = 0.01
        self._time = 0.01

        self._fps_duration = 3

        self._fps_frame_count = 0
        self._fps_last_measure = now
        self._fps = 0.01
        self._fps_is_new = False

    def update(self):
        """ Update timer on next frame """

        now = time.perf_counter()
        self._delta_time = now - self._now
        self._time = now - self._start
        self._now = now

        self._fps_frame_count += 1
        self._fps_is_new = False
        if self._now - self._fps_last_measure > self._fps_duration:
            self._fps = self._fps_frame_count / (self._now - self._fps_last_measure)
            self._fps_last_measure = self._now
            self._fps_frame_count = 0
            self._fps_is_new = True

    @property
    def fps_is_new(self) -> bool:
        """ Check if current FPS value is new """
        return self._fps_is_new

    @property
    def fps(self) -> float:
        """ Return average amount of frames per second """
        return self._fps

    @property
    def delta_time(self) -> float:
        """ Return duration between two last frames """
        return self._delta_time

    @property
    def time(self) -> float:
        """ Return time elapsed from timer initialization """
        return self._time


