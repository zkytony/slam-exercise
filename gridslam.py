import math
import numpy as np
from numpy.linalg import inv, LinAlgError
import sympy
import matplotlib.pyplot as plt
import sys


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

def eval_model(model, symbols, values, to_np=False, dtype=np.float64):
    if len(symbols) != len(values):
        raise ValueError("Cannot evaluate model. Symbols and values mismatch.")
    result = model.subs([(symbols[i], values[i]) for i in range(len(symbols))])
    if to_np:
        return np.array(result, dtype=dtype)
    else:
        return result

def mymul(a, b):
    res = np.matmul(a,b)
    res[np.isnan(res)] = 0
    return res

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
        self._belief_fig = None
        self.initialize_belief()
        self.initialize_models()

    @property
    def known_correspondence(self):
        return self._known_correspondence
    
    @property
    def motion_model(self):
        return self._motion_model

    @property
    def sensor_model(self):
        return self._sensor_model

    @property
    def sensor_params(self):
        return self._sensor_params

    @property
    def current_map(self):
        return self._mu[3:].reshape(-1,2), self._Sigma[3:,3:]

    @property
    def current_pose(self):
        return self._mu[:3], self._Sigma[:3,:3]
    
    def initialize_belief(self):
        # Initialize Gaussian belief
        state_dim = 3+2*self._num_landmarks
        self._mu = np.zeros((state_dim,))
        self._Sigma = np.zeros((state_dim, state_dim))
        self._Sigma[3:,3:].fill(sys.float_info.max)

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
                [0, self._motion_params['sigma_y']**2, 0],
                [0, 0, self._motion_params['sigma_th']**2]
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

    def plot_belief(self):
        
        m, cov_m = self.current_map
        p, cov_p = self.current_pose
        
        m_x = [int(round(m[i][0])) for i in range(len(m))
               if not np.isnan(m[i][0]) and not np.isnan(m[i][1])]
        m_y = [int(round(m[i][1])) for i in range(len(m))
               if not np.isnan(m[i][1]) and not np.isnan(m[i][0])]
        print(m_x)
        print(m_y)
        p[np.isnan(p)] = 0
        p = [int(round(p[i])) for i in range(2)] + [p[2]]

        if self._belief_fig is None:
            plt.ion()
            self._belief_fig = plt.figure()
            ax = self._belief_fig.add_subplot(111)
            self._map_plot = ax.scatter(m_x, m_y, 200)
            self._robot_plot = ax.scatter([p[0]], [p[1]], 200)
            self._belief_fig.canvas.draw()
            self._belief_fig.canvas.flush_events()
        else:
            ax = self._belief_fig.axes[0]
            ax.clear()
            self._map_plot = ax.scatter(m_x, m_y, 200)
            self._robot_plot = ax.scatter([p[0]], [p[1]], 200)
            self._belief_fig.canvas.draw()
            self._belief_fig.canvas.flush_events()
        

    def update(self, u, z_withc):
        if self._known_correspondence:
            z, c = z_withc
            self._update_known_correspondence(u, z, c)
        else:
            z = z_withc
            raise ValueError("EKF SLAM with unknown correspondence is not implemented.")            

    def _update_known_correspondence(self, u, z, c):
        N = self._num_landmarks
        
        Fx = np.zeros((3, 3+2*N))
        Fx[:3,:3] = np.identity(3)

        # Prediction
        pred_mu, pred_Sigma = None, None
        if self._motion_model == MotionModel.ODOMETRY:
            pred_mu = np.copy(self._mu)
            pred_mu[:3] = pred_mu[:3] + eval_model(self._grd,
                                                   self._symbols_motion,
                                                   self._mu[:3].tolist() + list(u), to_np=True).reshape(-1,)
            Jgrd = eval_model(self._Jgrd,
                              self._symbols_motion,
                              self._mu[:3].tolist() + list(u), to_np=True) # equivalent to gt in Prob.Rob.pg.318
            Gt = np.identity(3+2*N) + np.matmul(np.matmul(Fx.T, Jgrd), Fx)
            pred_Sigma = np.copy(self._Sigma)
            pred_Sigma = np.matmul(mymul(Gt, pred_Sigma), Gt.T)\
                         + np.matmul(np.matmul(Fx.T, self._R), Fx)
        else:
            raise ValueError("Motion model not supported")

        # Correction
        landmarks_seen = set({})
        z = np.array(z)
        if self._sensor_model == SensorModel.RANGE_BEARING:
            rx, ry, rth = self.current_pose[0]  # robot pose
            for i in range(len(z)):
                j = c[i]
                if j not in landmarks_seen:
                    d, th = z[i]  # distance, bearing
                    mjx = rx + int(round(d * math.cos(rth + th)))
                    mjy = ry + int(round(d * math.sin(rth + th)))
                    pred_mu[3+j*2:3+j*2+2] = [mjx, mjy]  # initial expectation
                    landmarks_seen.add(j)

                mu_mj = pred_mu[3+j*2:3+j*2+2]
                zi_hat = eval_model(self._hj,
                                   self._symbols_sensor,
                                   self._mu[:3].tolist() + mu_mj.tolist(), to_np=True).reshape(-1,)
                Jhj = eval_model(self._Jhj,
                                 self._symbols_sensor,
                                 self._mu[:3].tolist() + mu_mj.tolist(), to_np=True)
                Fxj = np.zeros((5, 3+2*N))
                Fxj[:3,:3] = np.identity(3)
                Fxj[3:5,3+(2*j-2):3+(2*j-2)+2] = np.identity(2)

                Hi = np.matmul(Jhj, Fxj)

                try:
                    Ki = np.matmul(mymul(pred_Sigma, Hi.T),
                                   inv(np.matmul(mymul(Hi, pred_Sigma), Hi.T) + self._Q))
                    pred_mu = pred_mu + np.matmul(Ki, (z[i] - zi_hat))
                    pred_Sigma = mymul((np.identity(3+2*N) - np.matmul(Ki, Hi)), pred_Sigma)
                except LinAlgError as e:
                    print("Update failed. Inverse not computable")
            th = pred_mu[2]
            pred_mu[np.isnan(pred_mu)] = 0
            self._mu = np.array([int(round(pred_mu[i])) for i in range(len(pred_mu))], dtype=np.float64)
            self._mu[2] = th % (2*math.pi)
            self._Sigma = pred_Sigma
                                
        else:
            raise ValueError("Sensor model not supported")        
