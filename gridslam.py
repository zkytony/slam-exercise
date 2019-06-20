import math
import numpy as np


class SensorModel:
    RANGE_BEARING = 0
    RAY_CAST = 1

    @classmethod
    def interpret_params(cls, sensor_model, sensor_params):
        if sensor_model == SensorModel.RANGE_BEARING:
            if type(sensor_params) == dict:
                if not ('max_range' in sensor_params \
                        and 'min_range' in sensor_params\
                        and 'view_angles' in sensor_params):
                    raise ValueError("Invalid sensor parameters for range sensor %s" % sensor_params)
                else:
                    return sensor_params
            elif type(sensor_params) == tuple:
                max_range, min_range, view_angles = sensor_params
                return {'max_range': max_range,
                        'min_range': min_range,
                        'view_angles': view_angles}

            
class MotionModel:
    ODOMETRY = 0
    VELOCITY = 1


class EKFSlamRobot:

    def __init__(self, num_landmarks,
                 sensor_model=SensorModel.RANGE_BEARING, sensor_params={},
                 motion_model=MotionModel.ODOMETRY, motion_params={}):
        self._sensor_model = sensor_model
        self._sensor_params = sensor_params
        self._motion_model = motion_model
        self._motion_params = motion_params

        # Initialize Gaussian belief
        self._pr = np.array([0, 0, 0])           # robot pose
        self._pm = np.zeros((num_landmarks*2,))  # landmark poses
        self._cov_rr = np.zeros((3,3))
        self._cov_rm = np.zeros((3,2*num_landmarks))
        self._cov_mr = np.zeros((2*num_landmarks,3))
        self._cov_mm = np.full((2*num_landmarks,2*num_landmarks), float('inf'))


    @property
    def est_pose(self):
        ex, ey, eth = self._pr
        ex = int(round(ex))
        ey = int(round(ey))
        eth = eth % (2*math.pi)        
        return (ex, ey, eth)

    @property
    def motion_model(self):
        return self._motion_model
    
    @property
    def sensor_model(self):
        return self._sensor_model
    
    @property
    def sensor_params(self):
        return self._sensor_params

    def update(self, u, z):
        print("### Observation:")
        print(z)
        print("### Control:")
        print(u)
        print("-------Update--------")
        print("pr (t-1): %s" % self._pr)
        
        # Prediction
        if self._motion_model == MotionModel.ODOMETRY:
            drot1, dtrans, drot2 = u
            rx, ry, rth = self._pr
            self._pr = np.array([int(round(rx + dtrans*math.cos(rth + drot1))),
                                 int(round(ry + dtrans*math.sin(rth + drot1))),
                                 (rth + drot1 + drot2) % (2*math.pi)])

        














    # def observe(self, z):
    #     """
    #     Receives observation z from the sensor.
    #     Sensor model: range-bearing model (i.e. beam orientation and distance)"""
    #     print("### Observation:")
    #     print(z)
    #     self._zt = z
    #     # if self._sensor_model == SensorModel.RANGE_BEARING:
    #     #     pass

    # def move(self, u):
    #     """
    #     Receives control input u. 
    #     Motion model: Odometry model"""
    #     print("### Control:")
    #     print(u)
    #     self._ut = u
    #     # if self._motion_model == MotionModel.ODOMETRY:
    #     #     drot1, dtrans, drot2 = u
        
