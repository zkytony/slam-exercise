import math
import numpy as np
import sympy


class SensorModel:
    RANGE_BEARING = 0
    RAY_CAST = 1

    @classmethod
    def interpret_params(cls, sensor_model, sensor_params):
        if sensor_model == SensorModel.RANGE_BEARING:
            required_params = set({
                'max_range', 'min_range', 'view_angles',
                'sigma_dist', 'sigma_bearing'
            })
            if type(sensor_params) == dict:
                if sensor_params.keys() == required_params:
                    return sensor_params
            else:
                raise ValueError("parameters must be a dict!")

            
class MotionModel:
    ODOMETRY = 0
    VELOCITY = 1
    
    @classmethod
    def interpret_params(cls, motion_model, motion_params):
        if motion_model == MotionModel.ODOMETRY:
            required_params = set({
                'sigma_x', 'sigma_y', 'sigma_th'
            })
            if type(motion_params) == dict:
                if motion_params.keys() == required_params:
                    return motion_params
            else:
                raise ValueError("parameters must be a dict!")    


class EKFSlamRobot:

    def __init__(self, num_landmarks,
                 sensor_model=SensorModel.RANGE_BEARING, sensor_params={},
                 motion_model=MotionModel.ODOMETRY, motion_params={},
                 known_correspondence=True):
        """
        num_landmarks (int): must be provided if 'known_correspondence'.
            If unknown, then -1.
        """
        self._num_landmarks = num_landmarks
        self._sensor_model = sensor_model
        self._sensor_params = SensorModel.interpret_params(sensor_model, sensor_params)
        self._motion_model = motion_model
        self._motion_params = MotionModel.interpret_params(motion_model, motion_params)        
        self._known_correspondence = True

    @property
    def state_dim(self):
        """dimensionality of state"""
        return 3 + 2 * self._num_landmarks  # robot: (x,y,th); landmark i: (x,y)

    @property
    def cov_rr(self):
        return self._Sigma[:3, :3]

    @property
    def cov_rm(self):
        return self._Sigma[:3, 3:]

    @property
    def cov_mr(self):
        return self._Sigma[3:, :3]

    @property
    def cov_mm(self):
        return self._Sigma[3:, 3:]

    @property
    def est_robot_pose(self):
        return self._mu[:3]

    def est_landmark_pose(self, i):
        return self._mu[3+i*2:3+i*2+2]

    @property
    def known_correspondence(self):
        return self._known_correspondence
    
    def initialize_belief(self):
        # Initialize Gaussian belief
        self._mu = np.zeros((self.state_dim,))
        self._Sigma = np.zeros((self.state_dim, self._state_dim))
        self.cov_mm.fill(float('inf'))


    def initialize_models(self):
        if self._motion_model == MotionModel.ODOMETRY:
            x, y, th, drot1, dtrans, drot2 = sympy.symbols('x y th dr1 dt dr2')
            state_robot = sympy.Matrix([x,y,th])
            self._g = sympy.Matrix([
                x + dtrans * sympy.cos(th + drot1),
                y + dtrans * sympy.sin(th + drot1),
                th + drot1 + drot2
            ])
            self._G = self._g.jacobian(state_robot)
            self._R = np.array([
                [self._motion_params['sigma_x']**2, 0, 0],
                0, [self._motion_params['sigma_y']**2, 0],
                0, 0, [self._motion_params['sigma_th']**2]
            ])

        if self._sensor_model == SensorModel.RANGE_BEARING:
            x, y, th = sympy.symbols('x y th')   # robot pose
            mjx, mjy = sympy.symbols('mjx mjy')  # pose of corresponding landmark j
            self._hi = sympy.Matrix([
                sympy.sqrt((mjx - x)**2 + (mjy - y)**2),
                sympy.atan2(mjy - y, mjx - x) - th
            ])
            state = sympy.Matrix([x,y,th,mjx,mjy])
            self._Hi = self._hi.jacobian(state)
            self._Q = np.array([[self._sensor_params['sigma_dist']**2, 0],
                                [0, self._sensor_params['sigma_bearing']**2]])
        

    def update(self, u, z_withc):
        if self._known_correspondence:
            z, c = z_withc
            self._update_known_correspondence(u, z, c)
        else:
            z = z_withc
            raise ValueError("EKF SLAM with unknown correspondence is not implemented.")            

    def _update_known_correspondence(self, u, z, c):
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
        
