from deepxde.geometry import geometry_1d, geometry_2d, geometry_3d, geometry_nd
import numpy as np


class Interval(geometry_1d.Interval):
    def __init__(self, x_left: float, x_right: float):
        super().__init__(l=x_left, r=x_right)


class Rectangle(geometry_2d.Rectangle):
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        super().__init__(xmin=np.float32([x_min, y_min]), xmax=np.float32([x_max, y_max]))


class Triangle(geometry_2d.Triangle):
    def __init__(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
        super().__init__(x1=np.float32([x1, y1]), x2=np.float32([x2, y2]), x3=np.float32([x3, y3]))


class Polygon(geometry_2d.Polygon):
    def __init__(self, *points_2d: list):
        super().__init__(vertices=np.float32(points_2d))


class Disk(geometry_2d.Disk):
    def __init__(self, center_x: float, center_y: float, radius):
        super().__init__(center=np.float32([center_x, center_y]), radius=radius)


class Cuboid(geometry_3d.Cuboid):
    def __init__(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
        super().__init__(xmin=np.float32([x1, y1, z1]), xmax=np.float32([x2, y2, z2]))


class Sphere(geometry_3d.Sphere):
    def __init__(self, center_x: float, center_y: float, center_z: float, radius):
        super().__init__(center=np.float32([center_x, center_y, center_z]), radius=radius)


class HyperPlane(geometry_nd.Hypercube):
    def __init__(self, xmin_nd: list, xmax_nd: list):
        super().__init__(xmin=np.float32(xmin_nd), xmax=np.float32(xmax_nd))


class HyperSphere(geometry_nd.Hypersphere):
    def __init__(self, center_nd: list, radius: float):
        super().__init__(center=np.float32(center_nd), radius=np.float32(radius))
