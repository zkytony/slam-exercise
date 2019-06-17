# Defines basic classes including:
# World, Environment, Space, Distribution, Agent

from abc import ABC, abstractmethod

class Space:

    """A space is a set of entities. Generally a multi-dimensional
    space is a set of tuples, where each element is of a certain type.
    For example, a 2D discrete space is a set of (x,y) pairs, where x
    and y are both ints."""

    def __init__(self, obj_types):
        """
        objtypes (tuple): a tuple of strings/classes that indicate the element type at
            each dimension of the space.
        """
        self._obj_types = obj_types

    def contains(self, e):
        """
        `e` (either tuple or an object): element that could be in the space.
        """
        # By default, just checks if the element is of the desired type.
        return type(e) == self._obj_types

    def n_dim(self):
        """Dimensionality of this space"""
        if type(self._obj_types) == tuple:
            return len(self._obj_types)
        else:
            return 1

    def size(self):
        """If this is a multi-dimensionanl space, return a tuple of integers (could be
        infinity). Otherwise, return a single integer (or infinity)"""
        # by default return infinity.
        return float('inf')


class Distribution:

    """A distribution is a mapping from an entity in a spacec
    to a real value. If normalized, it is the probability."""

    # TODO: Most likely value?

    def __init__(self, space, val_func):
        """
        val_func(function): a function that maps an element in the space
        to a real value.
        """
        self._space = space
        self._val_func = val_func

    def val(self, e):
        if not self._space.contains(e):
            return 0
        else:
            return self._val_func(e)
    

class Object(ABC):
    
    def __init__(self, env, state_space,
                 action_space, motion_model, letter="O"):
        self._env = env
        self._letter = letter
        self._state_space = state_space
        self._action_space = action_space
        self._motion_model = motion_model

    def set_init_belief(self, init_belief):
        self._belief = init_belief

    @property
    def belief(self):
        return self._belief
    @property
    def state_space(self):
        return self._state_space

    def initialized(self):
        return self._belief is not None
        
    def take_action(self, a):
        if not action_space.contains(a):
            raise ValueError("Action %s is not valid." % a)
        # Here, observation is the  measurement z_t, obtained given that the
        # agent has taken an action a_t in the environment, which brings
        # the agent to state x_t (state estimation after control).
        # observation = self._env.agent_action(self, a)
        self._update_belief(a)#, observation)
        

    # ? What about the normalizing factor eta?
    # Note, different from P(o|s',a) in Kaebling, Littman POMDP paper,
    # here we have assumed that a_t induces x_t, instead of affecting
    # the 'next state'. This is in line with Probabilistic Robtics, and
    # they mean the same thing underneath, that is, the probability of
    # observing a measurement given a state where the observation is made.
    @abstractmethod
    def _update_belief(self, a):#, observation):
        # """Should be overwritten by specific methods such as Kalman Filter"""
        pass
        # # Bayesian filtering (vanilla)
        # self._belief \
        #     = self._sensor_model(observation, self._belief, a).multiply(
        #         Distribution.Integrate(self._motion_model().multiply(self._belief),
        #                                self._state_space)
        
    
class Agent(Object):

    """
    This agent is probabilistic, meaning it holds only a distribution
    of its states rather than the true state.
    """

    def __init__(self, env, state_space, action_space,
                 observation_space,
                 sensor_model, motion_model, 
                 letter="R"):
        """
        env (Environment): the environment that the agent lives in,
           which describes how the world changes as the agent interacts with it.
        sensor_model (Distribution) defines p(z_t|x_t)
        motion model (Distribution) defines p(x_{t+1}|x_t,u_t)
        normalizing_factor (Distribution) defines 1/p(z_t|z_{1:t=1},u_{1:t})
        """
        super().__init__(env, state_space, action_space, motion_model,
                         letter=letter)
        self._observation_space = observation_space
        self._sensor_model = sensor_model

    @abstractmethod
    def observes(self):
        pass
    
class Environment(ABC):

    """
    An environment describes the underlying characteristics of a world,
    when an agent interacts with it. An environment does not depend
    on the agent, which means there could be multple agents behaving
    in the same environment simultaneously.

    The agent knows about the environment, but the environment does
    not know about the agent. I know Earth; Earth doesn't know me.
    """
    def  __init__(self):
        pass

    @property
    def world(self):
        return self._world

    @abstractmethod
    def size(self):
        pass

    def agent_act(self, agent, action):
        return self._provide_observation(agent, action)

    @abstractmethod
    def _provide_observation(self, agent, action):
        pass

