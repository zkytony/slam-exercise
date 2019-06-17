import numpy as np
from basics import Space





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
..x........x.
..x..xx......
......xxxxx..
"""

class Grid2D(Space):

    """
    a 2D grid, but with 3D information (because every cell has a value, binary). 
    The 3D information is, however, fixed at initialization.
    """

    def __init__(self, arr2d):
        self._d = arr2d
        self._w, self._h = len(arr2d[0]), len(arr2d)

    def contains(self, e):
        """
        `e` (either tuple or an object): element that could be in the space.
            Should be (x,y,V) according to our coordinate system.
        """
        if type(e) != tuple or len(e) != 2:
            return False
        elif type(e[0]) != int or type(e[1]) != int:
            return False
        elif (e[0] >= 0 and e[0] < self._w) \
             and (e[1] >= 0 and e[1] < self._h) \
             and (e[2] == 0 or e[2] == 1):
            return True
        else:
            return False

    def dimensionality(self):
        return 3

    def enumerate(self, rnge=None):
        if len(rnge) != self.dimensionality():
            raise ValueError("Invalidd range dimensionality.")
        if rnge is None:
            rnge = ((0, self._w),(0, self._h))
        rgx, rgy = rnge
        for y in range(rgy[0], rgy[1]):
            for x in range(rgx[0], rgx[1]):
                for v in range(1):
                    yield (x, y, v)

    def has_subset(self, other):
        """from stackoverflow: https://stackoverflow.com/questions/37261709/numpy-check-if-2d-array-is-subset-of-2d-array"""
        return np.in1d(b.ravel(), a.ravel()).all()

    @classmethod
    def from_string(cls, worldstr):
        lines = [l for l in worldstr.splitlines()
                 if len(l) > 0]
        w, h = len(lines[0]), len(lines)
        arr2d = np.empty((h,w), dtype=np.int32)
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
            except Exception:
                import pdb; pdb.set_trace()
        return Grid2D(arr2d)
        
    def to_str(self):
        arr2d = np.copy(self._d)
        # rx, ry = self._robotpose
        # arr2d[ry, rx] = 2
        worldstr = "\n".join(["".join(map(str,l))
                              .replace("0",".")
                              .replace("1","x")
                              for l in arr2d])
        return worldstr



    
if __name__ == "__main__":
    world = Grid2D.from_string(world1)
    print(world.to_str())
