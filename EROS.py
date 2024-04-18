import numpy as np
import cv2

class Surface:
    def __init__(self):
        self.kernel_size = 0
        self.half_kernel = 0
        self.parameter = 0.0
        self.time_now = 0.0
        # self.actual_region = None
        self.surf = np.matrix([])

    def get_surface(self):
        pass

    def init(self, width, height, kernel_size=5, parameter=0.0):
        pass

    def update(self, x, y, ts, p):
        pass

    def temporal_decay(self, ts, alpha):
        pass

    def spatial_decay(self, k):
        pass


class EROS(Surface):
    def update(self, x, y, t=0, p=0):
        odecay = self.parameter**(1.0 / self.kernel_size)
        self.surf[y:y+self.kernel_size, x:x+self.kernel_size] *= odecay
        self.surf[y+self.half_kernel, x+self.half_kernel] = 255.0