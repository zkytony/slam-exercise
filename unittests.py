from core import *
import random
from grid2d import GridWorld, world1, GridAgent

def test_space():
    int_space = Space(int)
    assert int_space.contains(1)
    assert int_space.contains(-1)
    assert int_space.contains(0)
    for i in range(100):
        assert int_space.contains(random.randint(-10000, 10000))

    assert int_space.n_dim() == 1

def test_distribution():
    int_space = Space(int)
    int_distr = Distribution(int_space, lambda x: 0.0)
    assert int_distr.val(10) == 0

def test_environment():
    # Use the gridworld
    world = GridWorld.from_string(world1)
    print(world.to_str())

def test_agent():
    # Use the gridworld agent
    world = GridWorld.from_string(world1)
    agent = GridAgent(world)
    init_belief = Distribution(agent.state_space,
                               lambda x: 1.0 if x == (0,0) else 0.0)
    agent.set_init_belief(init_belief)
    assert agent.belief.val((0,0)) == 1.0
    assert agent.belief.val((5,5)) == 0.0

if __name__ == "__main__":
    test_space()
    test_distribution()
    test_environment()
    test_agent()
    print("passed.")
