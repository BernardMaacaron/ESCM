import numpy as np

def update_surface(surf, x, y, kernel_size, half_kernel):
    surf[y:y+kernel_size, x:x+kernel_size] = 0
    surf[y+half_kernel, x+half_kernel] = 255.0
