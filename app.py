import numpy as np
import pygame
import cv2
import math
import random
from gridslam import MotionModel, SensorModel, EKFSlamRobot

# Coordinates
# ---> X
# |
# v
# Y
# Angle: 0 is right (x-axis positive), counter-clockwise.
world1 = \
"""
..........x..
.........xx..
...xx.....x..
......xx.....
..x........x.
..x....R...x.
..x..xx......
......xxxxx..
"""

world2 = \
"""
..............
..x..x..x..x..
..............
..x..x..x..x..
R.............
"""

world3 = \
"""
..........
.xxxxxxxx.
.xR.....x.
.x..xx..x.
.x......x.
.xxxxxxxx.
..........
"""

world4= \
"""
............................
..xxxxxxxxxxxxxxxxxxxxxxxx..
..xR.....................x..
..x......................x..
..x..xxxxxxxxxxxxxxxxxx..x..
..x..x................x..x..
..x..xxxxxxxxxxxxxxxxxx..x..
..x......................x..
..x......................x..
..xxxxxxxxxxxxxxxxxxxxxxxx..
............................
"""

world5 = \
"""
.............
.xxxxxxxxxx..
.xR....x..x..
.x..x..x..x..
.x..x.....x..
.xxxxxxxxxx..
.............
"""

world6 = \
"""
....................................
....................................
.R..................................
....................................
....................................
.....x.......x.......x.......x......
....................................
....................................
....................................
....................................
....................................
....................................
....................................
.....x.......x.......x.......x......
....................................
....................................
....................................
....................................
....................................
""" # textbook

def huge_random_world(w=100, h=100, prob=0.01):
    world = ""
    robot_pose = (random.randint(0,w-1),
                  random.randint(0,h-1))
    res = 300 // max(w,h)
    for i in range(w):
        line = ["."] * h
        for j in range(h):
            if (i,j) == robot_pose:
                line[j] = "R"
            else:
                if random.uniform(0, 1/prob) < 1:
                    line[j] = "x"
        world += "".join(line) + "\n"
    return world, res

def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def inclusive_within(v, rg):
    a, b = rg # range
    return v >= a and v <= b

def get_coords(arr2d):
    rows = np.arange(arr2d.shape[0])
    cols = np.arange(arr2d.shape[1])
    coords = np.empty((len(rows), len(cols), 2), dtype=np.intp)
    coords[..., 0] = rows[:, None]
    coords[..., 1] = cols
    return coords.reshape(-1, 2)

class GridWorld:
    
    def __init__(self, worldstr):
        """
        r: resolution, number of pixels per cell width.
        """
        lines = [l for l in worldstr.splitlines()
             if len(l) > 0]
        w, h = len(lines[0]), len(lines)
        arr2d = np.zeros((w,h), dtype=np.int32)
        robotpose = None
        landmarks = set({})
        for y, l in enumerate(lines):
            if len(l) != w:
                raise ValueError("World size inconsistent."\
                                 "Expected width: %d; Actual Width: %d"
                                 % (w, len(l)))
            for x, c in enumerate(l):
                if c == ".":
                    arr2d[x,y] = 0
                elif c == "x":
                    arr2d[x,y] = 1
                    landmarks.add((x,y))
                elif c == "R":
                    arr2d[x,y] = 0
                    robotpose = (x,y,0)
        if robotpose is None:
            raise ValueError("No initial robot pose!")
        self._d = arr2d
        self._robotpose = robotpose
        self._landmarks = landmarks
        self._last_z = []
        
    @property
    def width(self):
        return self._d.shape[0]
    
    @property
    def height(self):
        return self._d.shape[1]

    @property
    def arr(self):
        return self._d

    @property
    def last_observation(self):
        return self._last_z

    @property
    def num_landmarks(self):
        return len(self._landmarks)

    @property
    def robotpose(self):
        # The pose is (x,y,th)
        return self._robotpose

    def valid_pose(self, x, y):
        if x >= 0 and x < self.width \
           and y >= 0 and y < self.height:
            return self._d[x,y] == 0
        return True

    def move_robot(self, forward, angle, motion_model):
        """
        forward: translational displacement (vt)
        angle: angular displacement (vw)
        """
        # First turn, then move forward.
        begin_pose = self._robotpose
        rx, ry, rth = begin_pose
        rth += angle
        rx = int(round(rx + forward*math.cos(rth)))
        ry = int(round(ry + forward*math.sin(rth)))
        rth = rth % (2*math.pi)

        if motion_model == MotionModel.ODOMETRY:
            if self.valid_pose(rx, ry):
                self._robotpose = (rx, ry, rth)
                rx0, ry0, rth0 = begin_pose
                dtrans = dist((rx, ry), (rx0, ry0))
                drot1 = (math.atan2(ry - ry0, rx - rx0) - rth0) % (2*math.pi)
                drot2 = (rth - rth0 - drot1) % (2*math.pi)
                return (drot1, dtrans, drot2)
            else:
                return (0, 0, 0)                    


    def provide_observation(self, sensor_model, sensor_params, known_correspondence=False):
        """Given the current robot pose, provide the observation z."""
        def in_field_of_view(th, view_angles):
            """Determines if the beame at angle `th` is in a field of view of size `view_angles`.
            For example, the view_angles=180, means the range scanner scans 180 degrees
            in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
            fov_right = (0, view_angles / 2)
            fov_left = (2*math.pi - view_angles/2, 2*math.pi)
            return inclusive_within(th, fov_left) or inclusive_within(th, fov_right) 
            
        params = SensorModel.interpret_params(sensor_model, sensor_params)
        if sensor_model == SensorModel.RANGE_BEARING:
            # TODO: right now the laser penetrates through obstacles. Fix this?
            rx, ry, rth = self._robotpose
            z_candidates = [(dist(l, (rx, ry)),  # distance
                             (math.atan2(l[1] - ry, l[0] - rx) - rth) % (2*math.pi), # bearing (i.e. orientation)
                             i)
                            for i,l in enumerate(self._landmarks)  #get_coords(self._d)
                            if dist(l, self._robotpose[:2]) <= params['max_range']\
                            and dist(l, self._robotpose[:2]) >= params['min_range']]

            z_withc = [(d, th, i) for d,th,i in z_candidates
                       if in_field_of_view(th, params['view_angles'])]
            z = [(d, th) for d, th, _ in z_withc]
            c = [i for _, _, i in z_withc]
            self._last_z = z
            
            if known_correspondence:
                return z, c
            else:
                return z
            
class Environment:

    def __init__(self, gridworld, robot, res=30, fps=30):
        """
        r: resolution, number of pixels per cell width.
        """
        self._gridworld = gridworld
        self._robot = robot
        self._resolution = res
        self._img = self._make_gridworld_image(res)
        
        
        self._running = True
        self._display_surf = None
        self._fps = fps
        self._playtime = 0

    def _make_gridworld_image(self, r):
        arr2d = self._gridworld.arr
        w, h = arr2d.shape
        img = np.full((w*r,h*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(h):
                if arr2d[x,y] == 0:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (255, 255, 255), -1)
                elif arr2d[x,y] == 1:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)                    
        return img

    @staticmethod
    def draw_robot(img, x, y, th, size, color=(255,12,12)):
        radius = int(round(size / 2))
        cv2.circle(img, (y+radius, x+radius), radius, color, thickness=2)

        endpoint = (y+radius + int(round(radius*math.sin(th))),
                    x+radius + int(round(radius*math.cos(th))))
        cv2.line(img, (y+radius,x+radius), endpoint, color, 2)

    @staticmethod
    def draw_observation(img, z, rx, ry, rth, r, size, color=(12,12,255)):
        radius = int(round(r / 2))
        for d, th in z:
            lx = rx + int(round(d * math.cos(rth + th)))
            ly = ry + int(round(d * math.sin(rth + th)))
            cv2.circle(img, (ly*r+radius,
                             lx*r+radius), size, (12, 12, 255), thickness=-1)

    def render_env(self, display_surf):
        # draw robot, a circle and a vector
        rx, ry, rth = self._gridworld.robotpose
        r = self._resolution  # Not radius!
        img = np.copy(self._img)
        Environment.draw_robot(img, rx*r, ry*r, rth, r, color=(255, 12, 12))
        Environment.draw_observation(img, self._gridworld.last_observation, rx, ry, rth, r, r//3, color=(12,12,255))
        pygame.surfarray.blit_array(display_surf, img)
        
    @property
    def img_width(self):
        return self._img.shape[0]
    
    @property
    def img_height(self):
        return self._img.shape[1]
 
    def on_init(self):
        # pygame init
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            u = None
            if event.key == pygame.K_LEFT:
                u = self._gridworld.move_robot(0, -math.pi/4, self._robot.motion_model)  # rotate left 45 degree
            elif event.key == pygame.K_RIGHT:
                u = self._gridworld.move_robot(0, math.pi/4, self._robot.motion_model)  # rotate left 45 degree
            elif event.key == pygame.K_UP:
                u = self._gridworld.move_robot(1, 0, self._robot.motion_model)

            if u is not None:
                z_withc = self._gridworld.provide_observation(SensorModel.RANGE_BEARING, self._robot.sensor_params,
                                                              known_correspondence=robot.known_correspondence)

                print("------Truth------")
                print("   control: %s" % str(u))
                print("robot pose: %s" % str(self._gridworld.robotpose))
                
                self._robot.update(u, z_withc)  # the robot updates its belief
                m, Sigma_m = self._robot.current_map
                p, Sigma_p = self._robot.current_pose
                print("------Belief------")
                print("===Map===")
                print(m)
                print("===Pose===")
                print(p)
                self._robot.plot_belief(disk_size=20)
            
    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0
    
    def on_render(self):
        # self._display_surf.blit(self._background, (0, 0))
        self.render_env(self._display_surf)
        
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        rx, ry, rth = self._gridworld.robotpose
        pygame.display.set_caption("(%.2f,%.2f,%.2f) %s" % (rx, ry, rth*180/math.pi,
                                                            fps_text))
        pygame.display.flip() 
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__" :
    world, res = huge_random_world(w=10, h=10, prob=0.1)
    
    gridworld = GridWorld(world6)
    robot = EKFSlamRobot(gridworld.num_landmarks,
                         sensor_params={
                             'max_range':12,
                             'min_range':1,
                             'view_angles': math.pi,
                             'sigma_dist': 0.01,
                             'sigma_bearing': 0.01},
                         motion_params={
                             'sigma_x': 0.01,
                             'sigma_y': 0.01,
                             'sigma_th': 0.01
                         })

    theEnvironment = Environment(gridworld, robot, res=res, fps=20)
    theEnvironment.on_execute()

