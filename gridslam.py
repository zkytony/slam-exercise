import math
import numpy as np
from numpy.linalg import inv
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

def eval_model(model, symbols, values, np=False, dtype=np.float64):
    if len(symbols) != len(values):
        raise ValueError("Cannot evaluate model. Symbols and values mismatch.")
    result = model.subs([(symbols[i], values[i]) for i in range(len(symbols))])
    if np:
        return np.array(result, dtype=dtype)
    else:
        return result

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
    def known_correspondence(self):
        return self._known_correspondence

    def current_map(self):
        pass

    def current_pose(self):
        pass
    
    def initialize_belief(self):
        # Initialize Gaussian belief
        state_dim = 3+2*self._num_landmarks
        self._mu = np.zeros((state_dim,))
        self._Sigma = np.zeros((state_dim, state_dim))
        self.cov_mm.fill(float('inf'))


    def initialize_models(self):
        if self._motion_model == MotionModel.ODOMETRY:
            x, y, th, drot1, dtrans, drot2 = sympy.symbols('x y th dr1 dt dr2')
            state_robot = sympy.Matrix([x,y,th])
            self._grd = sympy.Matrix([
                dtrans * sympy.cos(th + drot1),
                dtrans * sympy.sin(th + drot1),
                drot1 + drot2
            ])
            self._Jgrd = self._grd.jacobian(state_robot)
            self._R = np.array([
                [self._motion_params['sigma_x']**2, 0, 0],
                0, [self._motion_params['sigma_y']**2, 0],
                0, 0, [self._motion_params['sigma_th']**2]
            ])
            self._symbols_motion = (x, y, th, drot1, dtrans, drot2)

        if self._sensor_model == SensorModel.RANGE_BEARING:
            x, y, th = sympy.symbols('x y th')   # robot pose
            mjx, mjy = sympy.symbols('mjx mjy')  # pose of corresponding landmark j
            self._hj = sympy.Matrix([
                sympy.sqrt((mjx - x)**2 + (mjy - y)**2),
                sympy.atan2(mjy - y, mjx - x) - th
            ])
            state = sympy.Matrix([x,y,th,mjx,mjy])
            self._Jhj = self._hj.jacobian(state)
            self._Q = np.array([[self._sensor_params['sigma_dist']**2, 0],
                                [0, self._sensor_params['sigma_bearing']**2]])
            self._symbols_sensor = (x, y, th, mjx, mjy)
        

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

        N = self._num_landmarks
        
        Fx = np.zeros((3, 3+2*n))
        Fx[:3,:3] = np.identity(3)

        # Prediction
        pred_mu, pred_Sigma = None, None
        if self._motion_model == MotionModel.ODOMETRY:
            pred_mu = np.copy(self._mu)
            pred_mu[:3] = pred_mu[:3] + eval_model(self._grd,
                                                   self._symbols_motion,
                                                   self._mu[:3].tolist() + list(u), np=True)
            Jgrd = eval_model(self._Jgrd,
                              self._symbols_motion,
                              self._mu[:3].tolist() + list(u), np=True) # equivalent to gt in Prob.Rob.pg.318
            Gt = np.identity(3+2*n) + Fx.T * Jgrd * Fx
            pred_Sigma = np.copy(self._Sigma)
            pred_Sigma = Gt * pred_sigma * Gt.T + Fx.T * self._R * Fx
        else:
            raise ValueError("Motion model not supported")

        # Correction
        landmarks_seen = set({})
        z = np.array(z)
        if self._sensor_model == SensorModel.RANGE_BEARING:
            rx, ry, rth = self.mu_r  # robot pose
            for i in range(len(z)):
                j = c[i]
                if j not in landmarks_seen:
                    d, th = z  # distance, bearing
                    mjx = rx + int(round(d * math.cos(rth + th)))
                    mjy = ry + int(round(d * math.sin(rth + th)))
                    pred_mu[3+j*2:3+j*2+2] = [mjx, mjy]  # initial expectation
                    landmarks_seen.add(j)

                mu_mj = pred_mu[3+j*2:3+j*2+2]
                z_hat = eval_model(self._hj,
                                   self._symbols_sensor,
                                   self._mu[:3].tolist() + mu_mj.tolist(), np=True)
                Jhj = eval_model(self._Jhj,
                                 self._symbols_sensor,
                                 self._mu[:3].tolist() + mu_mj.tolist(), np=True)
                Fxj = np.zeros((5, 3+2*N))
                Fxj[:3,:3] = np.identity(3)
                Fxj[3+(2*j-2):3+(2*j-2)+2,3:5] = np.identity(2)
                
                Hi = Jhj * Fxj

                Ki = pred_Sigma * Hi.T * inv(Hi * pred_Sigma * Hi.T + self._Q)
                pred_mu = pred_mu + Ki * (z - z_hat)
                pred_Sigma = (np.identity(3+2*N) - Ki * Hi) * pred_Sigma
            self._mu = pred_mu
            self._Sigma = pred_Sigma
                                
        else:
            raise ValueError("Sensor model not supported")        
