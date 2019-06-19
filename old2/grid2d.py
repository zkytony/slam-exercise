from abc import ABC, abstractmethod
import numpy as np
from basics import Space, Environment, Agent

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
    a 2D grid with certain size, within certain coordinate range.
    """
    def __init__(self, x_range, y_range):
        self._xmin, self._xmax = x_range
        self._ymin, self._ymax = y_range
        self._w = self._xmax - self._xmin
        self._h = self._ymax - self._ymin
        
    def contains(self, e):
        """
        `e` (either tuple or an object): element that could be in the space.
            Should be (x,y) according to our coordinate system.
        """
        if type(e) != tuple or len(e) != 2:
            return False
        elif type(e[0]) != int or type(e[1]) != int:
            return False
        elif (e[0] >= self._xmin and e[0] < self._xmax) \
             and (e[1] >= self._ymin and e[1] < self._ymax):
            return True
        else:
            return False

    def dimensionality(self):
        return 2

    def enumerate(self, rnge=None):
        if len(rnge) != self.dimensionality():
            raise ValueError("Invalid range dimensionality.")
        if rnge is None:
            rnge = ((0, self._w),(0, self._h))
        rgx, rgy = rnge
        for y in range(rgy[0], rgy[1]):
            for x in range(rgx[0], rgx[1]):
                yield (x, y)

    def has_subset(self, other):
        return other._xmin >= self._xmin and other._xmax <= self._xmax\
            and other._ymin >= self._ymin and other._ymax <= self.y_max

    
class GridAgentActionSpace(Space):
    """The agent can only move locally relative to its current
        position, in four directions, one cell at a time."""    
    def __init__(self):
        self._actions = ["N","E","S","W"]

    def contains(self, e):
        if type(e) == tuple:
            if len(e) != 1:
                return False
            e = e[0]

        if type(e) == str:
            for a in self._actions:
                if e[0] == a:
                    return True
        return False

    def dimensionality(self):
        return 1

    def enumerate(self):
        for a in self._actions:
            yield a

    def has_subset(self, other):
        return set(other._actions).issubset(set(self._actions))


class RangeObservationSpace(Space):
    """
    Suppose the agent is equpped with a range sensor that
    has a 180 degree field of reception. This means that
    the agent observes relative distances of landmarks.
    How many landmarks can the agent observe simultaneously
    depends on the "resolution" (deg/beam) of the sensor. This is a
    continuous observation space.
    """    
    def __init__(self, resolution, max_range, min_range):
        """
        `resolution`: number of degrees per range measuremnet
        """
        self._resolution = resolution
        self._max_range = max_range
        self._min_range = min_range
        self._n_beams = 180 / self._resolution

    def is_continuous(self):
        return True

    def dimensionality(self):
        return self._n_beams

    def contains(self):
        if type(e) != tuple or len(e) != self.dimensionality():
            return False
        for val in e:
            if not (type(val) == float and \
                    val <= self._max_range and val >= self._min_range):
                return False
        return True

    def enumerate(self, rnge=None):
        # It is impossible to enumerate over the observations space
        # because it is continuous.
        return False

    
class GridAgent(Agent):
    def __init__(self, gridworld,
                 s_res, s_maxrange, s_minrange, t0=0):
        self._state_space = Grid2D((0, gridworld._w), (0, gridworld._h))
        self._action_space = GridAgentActionSpace()
        self._observation_space = RangeObservationSpace(resolution, 5, 1)
        self._history = []  # (s, a, o)
        self._t0 = t0
        self._t = t0

    def observation(self, t):
        if t < self._t or\
           t >= self._t + len(self._history):
            raise ValueError("observation is lost/not obtained for time %d" % t)
        return self._history[t][2]

    def state(self, t):
        if t < self._t or\
           t >= self._t + len(self._history):
            raise ValueError("state is lost/not obtained for time %d" % t)
        return self._history[t][0]
    
    def action(self, t):
        if t < self._t or\
           t >= self._t + len(self._history):
            raise ValueError("state is lost/not obtained for time %d" % t)
        return self._history[t][1]

    @abstractmethod
    def take_action(self, env, a):
        pass

    @abstractmethod
    def observe(self, env, t):
        pass

    @abstractmethod
    def obtain_control(self, env, t):
        """Returns an action given current environment"""
        pass

    @abstractmethod
    def update(self, a, z, t):
        """Based on the current action and observation, update the belief of state"""
        pass


class GridWorld(Environment):

    """
    a 2D grid, but with 3D information (because every cell has a value, binary). 
    The 3D information is, however, fixed at initialization.
    """

    def __init__(self, arr2d):
        self._d = arr2d
        self._w, self._h = len(arr2d[0]), len(arr2d)

    def state(self, t):
        return self._d

    def action(self, t):
        return None

    def take_action(self, a):
        raise NotImplemented("Grid world is fixed and no action can be taken by itself.")

    def receive(self, agent, a, t):
        # The world does not change no matter what action the agent takes
        return True

    def internal_loop(self):
        # Nothing is going on in the static 2D grid
        return True

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
        return GridWorld(arr2d)
        
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
    world = GridWorld.from_string(world1)
    print(world.to_str())
