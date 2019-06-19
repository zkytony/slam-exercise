import numpy as np
import pygame
import cv2
import math
from pygame.locals import *

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
..x........x.
..x..xx..R...
......xxxxx..
"""

class GridWorld:
    
    def __init__(self, worldstr, r=30):
        """r: resolutoin, number of pixels per cell width"""
        lines = [l for l in worldstr.splitlines()
             if len(l) > 0]
        w, h = len(lines[0]), len(lines)
        arr2d = np.zeros((w,h), dtype=np.int32)
        img = np.full((w*r,h*r,3), 255, dtype=np.int32)
        robotpose = None
        for y, l in enumerate(lines):
            if len(l) != w:
                raise ValueError("World size inconsistent."\
                                 "Expected width: %d; Actual Width: %d"
                                 % (w, len(l)))
            for x, c in enumerate(l):
                if c == ".":
                    arr2d[x,y] = 0
                    img[x*r:x*r+r,y*r:y*r+r] = np.array([255, 255, 255])
                elif c == "x":
                    arr2d[x,y] = 1
                    img[x*r:x*r+r,y*r:y*r+r] = np.array([40, 31, 3])
                elif c == "R":
                    arr2d[x,y] = 0
                    img[x*r:x*r+r,y*r:y*r+r] = np.array([255, 255, 255])
                    robotpose = (x,y,0)
        if robotpose is None:
            raise ValueError("No initial robot pose!")
        self._d = arr2d
        self._img = img
        self._resolution = r
        self._robotpose = robotpose
        
    @property
    def width(self):
        return self._d.shape[0]
    
    @property
    def height(self):
        return self._d.shape[1]

    @property
    def img_width(self):
        return self._img.shape[0]
    
    @property
    def img_height(self):
        return self._img.shape[1]

    def render(self, display_surf):
        # draw robot, a circle and a vector
        rx, ry, rth = self._robotpose
        r = self._resolution  # Not radius!
        cv2.circle(self._img, (ry*r,rx*r), r//2, (255, 12, 12), thickness=1)

        endpoint = (ry*r + int(round(r/2*math.sin(rth))),
                    rx*r + int(round(r/2*math.cos(rth))))
        cv2.line(self._img, (ry*r,rx*r), endpoint, (255, 12, 12))
                 
        pygame.surfarray.blit_array(display_surf, self._img)
        

class App:

    def __init__(self, gridworld, fps=30):
        """resolution: number of pixels in the display per cell"""
        self._gridworld = gridworld
        
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._fps = fps
        self._playtime = 0
 
    def on_init(self):
        # pygame init
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self._gridworld.img_width,
                                                      self._gridworld.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True
 
    def on_event(self, event):
        if event.type == QUIT:
            self._running = False
            
    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0
    
    def on_render(self):
        # self._display_surf.blit(self._background, (0, 0))
        self._gridworld.render(self._display_surf)
        
        text = "FPS: {0:.2f}   Playtime: {1:.2f}".format(self._clock.get_fps(),
                                                         self._playtime)
        # text_surf = self._myfont.render(text, False, (255, 200, 255))
        # self._display_surf.blit(text_surf, (0, 0))
        pygame.display.set_caption(text)
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
    theApp = App(gridworld, fps=60)
    theApp.on_execute()
