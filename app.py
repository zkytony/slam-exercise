import numpy as np
import pygame
import cv2
import math
from gridslam import MotionModel, SensorModel, EKFSlamRobot

# Coordinates
# ---> X
# |
# v
# Y
# Angle: 0 is up, counter-clockwise.
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

def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

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
                    robotpose = (x,y,math.pi/4)
        if robotpose is None:
            raise ValueError("No initial robot pose!")
        self._d = arr2d
        self._robotpose = robotpose
        self._landmarks = landmarks
        
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


    def provide_observation(self, sensor_model, sensor_params):
        """Given the current robot pose, provide the observation z."""
        params = SensorModel.interpret_params(sensor_model, sensor_params)
        if sensor_model == SensorModel.RANGE_BEARING:
            landmarks_in_range = set(l for l in self._landmarks
                                     if dist(l, self._robotpose[:2]) <= params['max_range']\
                                     and dist(l, self._robotpose[:2]) >= params['min_range'])
            rx, ry, rth = self._robotpose
            z = [(dist(l, (rx, ry)),  # distance
                  math.atan2(l[1] - ry, l[0] - rx) - rth)  # bearing (i.e. orientation)
                 for l in landmarks_in_range]
            return z
            
class App:

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
        self._image_surf = None
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
        
    def render_env(self, display_surf):
        # draw robot, a circle and a vector
        rx, ry, rth = self._gridworld.robotpose
        r = self._resolution  # Not radius!
        radius = int(round(r / 2))
        img = np.copy(self._img)
        cv2.circle(img, (ry*r+radius, rx*r+radius), radius, (255, 12, 12), thickness=2)

        endpoint = (ry*r+radius + int(round(r/2*math.sin(rth))),
                    rx*r+radius + int(round(r/2*math.cos(rth))))
        cv2.line(img, (ry*r+radius,rx*r+radius), endpoint, (255, 12, 12), 2)
                 
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
            
            if event.key == pygame.K_LEFT:
                u = self._gridworld.move_robot(0, -math.pi/4, robot.motion_model)  # rotate left 45 degree
                robot.move(u)
            elif event.key == pygame.K_RIGHT:
                u = self._gridworld.move_robot(0, math.pi/4, robot.motion_model)  # rotate left 45 degree
                robot.move(u)
            elif event.key == pygame.K_UP:
                u = self._gridworld.move_robot(1, 0, robot.motion_model)
                robot.move(u)
                
            z = self._gridworld.provide_observation(SensorModel.RANGE_BEARING, robot.sensor_params)
            robot.observe(z)
                
            
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
    gridworld = GridWorld(world1)
    robot = EKFSlamRobot(sensor_params={'max_range':5, 'min_range':1})
    theApp = App(gridworld, robot, res=30, fps=60)
    theApp.on_execute()
