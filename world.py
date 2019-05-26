# 2D world
import numpy as np

# Coordinates
# ---> X
# |
# v
# Y
world1 = \
"""
..........x..
.........xx..
...xx.....x..
......xx.....
..x........x.
..x.....R..x.
..x..xx......
......xxxxx..
"""

class GridWorld:
    
    def __init__(self, arr2d, robot_init_pose):
        self._d = arr2d
        self._robotpose = robot_init_pose

    @classmethod
    def from_string(cls, worldstr):
        lines = [l for l in worldstr.splitlines()
                 if len(l) > 0]
        w, h = len(lines[0]), len(lines)
        arr2d = np.empty((h,w), dtype=np.int32)
        robot_init_pose = None
        for y, l in enumerate(lines):
            if len(l) != w:
                raise ValueError("World size inconsistent."\
                                 "Expected width: %d; Actual Width: %d"
                                 % (w, len(l)))
            try:
                for x, c in enumerate(l):
                    if c == ".":
                        arr2d[y,x] = 0
                    elif c == "x":
                        arr2d[y,x] = 1
                    elif c == "R":
                        arr2d[y,x] = 0
                        robot_init_pose = (x,y)
            except Exception:
                import pdb; pdb.set_trace()
        return GridWorld(arr2d, robot_init_pose)
        
    def to_str(self):
        arr2d = np.copy(self._d)
        rx, ry = self._robotpose
        arr2d[ry, rx] = 2
        worldstr = "\n".join(["".join(map(str,l))
                              .replace("0",".")
                              .replace("1","x")
                              .replace("2","R")
                              for l in arr2d])
        return worldstr


if __name__ == "__main__":
    world = GridWorld.from_string(world1)
    print(world.to_str())
        
