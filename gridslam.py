import numpy as np


class SensorModel:
    RANGE_BEARING = 0
    RAY_CAST = 1

    @classmethod
    def interpret_params(cls, sensor_model, sensor_params):
        if sensor_model == SensorModel.RANGE_BEARING:
            if type(sensor_params) == dict:
                if not ('max_range' in sensor_params \
                        and 'min_range' in sensor_params):
                    raise ValueError("Invalid sensor parameters for range sensor %s" % sensor_params)
                else:
                    return sensor_params
            elif type(sensor_params) == tuple:
                n_beams, max_range, min_range, angular_res = sensor_params
                return {'max_range': max_range,
                        'min_range': min_range}

            
class MotionModel:
    ODOMETRY = 0
    VELOCITY = 1


class EKFSlamRobot:

    def __init__(self,
                 sensor_model=SensorModel.RANGE_BEARING, sensor_params={},
                 motion_model=MotionModel.ODOMETRY, motion_params={}):
        self._sensor_model = sensor_model
        self._sensor_params = sensor_params
        self._motion_model = motion_model
        self._motion_params = motion_params
        self._estpose = (0, 0, 0)

    @property
    def motion_model(self):
        return self._motion_model
    
    @property
    def sensor_model(self):
        return self._sensor_model
    
    @property
    def sensor_params(self):
        return self._sensor_params

    def observe(self, z):
        """
        Receives observation z from the sensor.
        Sensor model: range-bearing model (i.e. beam orientation and distance)"""
        print("Observation:")
        print(z)
        if self._sensor_model == SensorModel.RANGE_BEARING:
            pass

    def move(self, u):
        """
        Receives control input u. 
        Motion model: Odometry model"""
        print("Control:")
        print(u)
        if self._motion_model == MotionModel.ODOMETRY:
            drot1, dtrans, drot2 = u
