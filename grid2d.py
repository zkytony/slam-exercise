# 2D world
import numpy as np
from core import Environment, Agent, Space, Distribution

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

################################
# 2D Grid Agent
################################
class GridAgent(Agent):

    # ------- State space -------- #
    class StateSpace(Space):
        """2D discrete gridworld state space"""
        def __init__(self, gridworld):
            self._gridworld = gridworld
            super().__init__((int,int))
            
        def contains(self, e):
            """
            `e` (either tuple or an object): element that could be in the space.
                Should be (x,y) according to our coordinate system.
            """
            if type(e) != tuple or len(e) != 2:
                return False
            elif type(e[0]) != int or type(e[1]) != int:
                return False
            elif (e[0] >= 0 and e[0] < self._gridworld._w) \
                 and (e[1] >= 0 and e[1] < self._gridworld._h):
                return True
            else:
                return False

        def size(self):
            return self._gridworld.size()

    # ------- Action space -------- #
    class ActionSpace(Space):
        """The agent can only move locally relative to its current
        position, in four directions, one cell at a time."""
        def __init__(self):
            super().__init__((str,))

        def contains(self, e):
            """
            `e` (either tuple or an object): element that could be in the space.
                Should be (x,y) according to our coordinate system.
            """
            if type(e) == tuple:
                if len(e) != 1:
                    return False
                e = e[0]

            if type(e) == str:
                if e[0] == "N" or e[0] == "E" or e[0] == "S" or e[0] == "W":
                    return True
            return False

        def size(self):
            return 4  #NESW
        
    
    # ------- Observation space -------- #
    class ObservationSpace(Space):
        """
        Suppose the agent is equpped with a range sensor that
        has a 180 degree field of reception. This means that
        the agent observes relative distances of landmarks.
        How many landmarks can the agent observe simultaneously
        depends on the "resolution" of the sensor. This is a
        continuous observation space.
        """
        def __init__(self, resolution, max_range, min_range):
            """
            `resolution`: number of degrees per range measuremnet
            """
            self._resolution = resolution
            self._max_range = max_range
            self._min_range = min_range
            obj_types = (float,) * int(round(180/self._resolution))
            super().__init__(obj_types)

        def size(self):
            return float('inf')

        def contains(self, e):
            if type(e) != tuple or len(e) != self.n_dim():
                return False
            for val in e:
                if not (type(val) == float and \
                        val <= self._max_range and val >= self._min_range):
                    return False
            return True


    # This is hard.
    class SensorModel(Distribution):
        def __init__(self):
            pass
    
    class MotionModel(Distribution):
        def __init__(self):
            pass        

    # ------- Grid Agent init -------
    def __init__(self, gridworld):
        super().__init__(gridworld,
                         GridAgent.StateSpace(gridworld),
                         GridAgent.ActionSpace(),
                         GridAgent.ObservationSpace(30, 5, 1),
                         GridAgent.SensorModel(),
                         GridAgent.MotionModel())

    def _update_belief(self, a):
        pass

    def observes(self):
        pass
        

################################
# 2D Grid World
################################
class GridWorld(Environment):

    """
    A 2D grid world is a 2-dimensional discrete integer-valued space,
    represented by a 2D array (np.ndarray). There is only one robot (agent)
    """
    
    def __init__(self, arr2d):
        self._d = arr2d
        self._w, self._h = len(arr2d[0]), len(arr2d)

    def size(self):
        return (self._w, self._h)

    # Implementing abstract method
    def _provide_observation(self, agent, action):
        pass
        
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
