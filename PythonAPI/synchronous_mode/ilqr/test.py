import numpy as np
from ilqr import ILQRController

data = [\
    {'waypoint': np.array([-117.09651947,  136.61997986, 0. , 0. , -1.29680216, 0.])}, \
    {'waypoint': np.array([-116.59664917, 136.6086731, 0. , 0. , -1.29680216, 0.])}, \
    {'waypoint': np.array([-116.09677887, 136.59735107, 0., 0., -1.29680216, 0.])}, \
    {'waypoint': np.array([-115.59690094, 136.58604431, 0., 0., -1.29680216, 0.])}, \
    {'waypoint': np.array([-115.09703064, 136.57472229, 0., 0., -1.29680216, 0.])}, \
    {'waypoint': np.array([-114.59716034, 136.56340027,  0., 0., -1.29680216, 0.])}, \
    {'measurements.t': np.array([-1.17103600e+02, 1.36307205e+02, -6.69517508e-03, -4.19127703e-01, 6.36095715e+00, 5.62562287e-01])}, \
    {'measurements.v': np.array([2.14972687e+00, 1.05032337e+00, 1.31203706e-05])}\
    ]


class loc:
    def __init__(self):
        self.x = 0.
        self.y = 0.
class rot:
    def __init__(self):
        self.yaw = 0.
class v:
    def __init__(self):
        self.x = 0.
        self.y = 0.

class trans:
    def __init__(self):
        self.location = loc()

class wp:
    def __init__(self):
        self.transform = trans() 

class t:
    def __init__(self):
        self.location = loc()
        self.rotation = rot()

class measure:
    def __init__(self):
        self.t = t()
        self.v = v()

future_wps = []
for i_wp in range(6): 
    future_wps.append(wp())
    future_wps[-1].transform.location.x = data[i_wp]['waypoint'][0]
    future_wps[-1].transform.location.y = data[i_wp]['waypoint'][1]
measurements = measure()
measurements.t.location.x = data[-2]['measurements.t'][0]
measurements.t.location.y = data[-2]['measurements.t'][1]
measurements.t.rotation.yaw = data[-2]['measurements.t'][2]
measurements.v.x = data[-1]['measurements.v'][0]
measurements.v.y = data[-1]['measurements.v'][1]


controller = ILQRController(target_speed = 1.0, steps_ahead = 10, dt = 0.1, half_width = 2.0)
controller.control(future_wps, measurements)



