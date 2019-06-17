
from abc import ABC, abstractmethod

class Space(ABC):

    """A space is a set of entities, each entity is a tuple of n dimensions,
    each either discrete or continuous. The space is either finite or infinite."""

    def __init__(self):
        pass

    @abstractmethod
    def contains(self, e):
        """
        Returns if e is an entity in this space
        """
        pass

    @abstractmethod
    def dimensionality(self):
        """
        Returns if e is an entity in this space
        """
        pass

    @abstractmethod
    def enumerate(self, func, args, d=None, rnge=None):
        """
        Enumerate all elements in this space, or enumerate all values along
        dimension `d`. Execute function `func` with `args` at every step.
        The enumerate is constrained with a given range `rnge`.
        """
        pass

    @abstractmethod
    def has_subset(self, other):
        """Returns True if `other` is a subset of the this space"""
        pass


class Distribution(ABC):
    
    """A distribution is a mapping from an entity in a space
    to a real value. Described as

    D: S -> R

    If normalized, it is a probability distribution
    over a space. """

    def __init__(self):
        pass

    @abstractmethod
    def integrate(self, rnge=None):
        """Integrate this distribution within range `rnge` over the space"""
        pass

    @abstractmethod
    def add(self, other):
        """add this dist. with another dist. over the same/subset space,
        resulting in a new distribution"""
        pass

    @abstractmethod
    def multiply(self, other):
        """multiply this dist. with another dist. over the same/subset space,
        resulting in a new distribution"""        
        pass
    
    @abstractmethod
    def max_k(self, k):
        """Returns the top k most likely entities in space"""
        pass

    @abstractmethod
    def sample(self, n=1):
        """Sample `n` entities based on the underlying probabilities"""
        pass

    @abstractmethod
    def normalize(self):
        """Returns a new distribution where it is guaranteed that the mapped values are
        normalized."""
        pass


class Agent(ABC):

    """An agent is a tuple (S, A, O, sensor_model, motion_model) capable of producing
    actions and observing in the environment."""
    
    def __init__(self):
        pass

    @abstractmethod
    def observation(self, t):
        pass

    @abstractmethod
    def state(self, t):
        pass
    
    @abstractmethod
    def action(self, t):
        pass

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


class Environment(ABC):

    """An environment is a tuple (S, A) where an agent can receive an observation and enforce
    an action upon."""

    def __init__(self):
        pass

    @abstractmethod
    def state(self, t):
        pass
    
    @abstractmethod
    def action(self, t):
        pass
    
    @abstractmethod
    def take_action(self, a):
        pass

    @abstractmethod
    def receive(self, agent, a, t):
        """Receives the action `a` performed by `agent`."""
        pass

    @abstractmethod
    def internal_loop(self):
        pass


class World:

    def __init__(self, agents, environment):
        self._agents = agents
        self._env = environment

    def loop(self, T=None):
        t = 0
        # Assume the agent and environment have been initialized at construction.
        while True if T is None else t < T:
            for agent in self._agents:
                a = agent.obtain_control(self._env, t)
                agent.take_action(env, a, t)
                self._env.receive(agent, a, t)
                self._env.internal_loop()
                z = agent.observe(env, t)
                agent.update_belief(a, z, t)
                t += 1
            
