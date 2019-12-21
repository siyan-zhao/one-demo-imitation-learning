import numpy as np
from kick_ass_utils import get_env

def test_sawyer_reach_env():
    env = get_env('sawyer_reach')
    env.reset()
    for j in range(2):
        env.reset()
        for i in range(300):
            env.step(np.random.randn(4))
            env.render()


def test_sawyer_touch_env():
    env = get_env('sawyer_touch')
    env.reset()
    for j in range(2):
        env.reset()
        for i in range(300):
            env.step(np.random.randn(4))
            env.render()


def test_cyl():
    env = get_env('cyl')
    env.reset()
    for j in range(2):
        env.reset()
        for i in range(300):
            env.step(np.random.randn(2))
            env.render()

def test_reacher():
    env = get_env('reacher')
    env.reset()
    for j in range(2):
        env.reset()
        for i in range(300):
            env.step(np.random.randn(2))
            env.render()


def test_reacher_multigoal():
    env = get_env('reacher_distractor')
    env.reset()
    for j in range(2):
        env.reset()
        for i in range(300):
            env.step(np.random.randn(2))
            env.render()


def test_pointmass():
    env = get_env('point')
    env.reset()
    for j in range(2):
        env.reset()
        for i in range(300):
            env.step(np.random.randn(2))
            env.render()


def test_pointmass_multigoal():
    env = get_env('point_disctractor')
    env.reset()
    for j in range(2):
        env.reset()
        for i in range(300):
            env.step(np.random.randn(2))
            env.render()


def main():
    test_sawyer_reach_env()
    test_sawyer_touch_env()
    test_cyl()
    test_pointmass()
    test_pointmass_multigoal()
    test_reacher()
    test_reacher_multigoal()


if __name__ == "__main__":
    main()
