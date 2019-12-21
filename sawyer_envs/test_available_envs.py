def test_available_envs():
    import gym
    gym.logger.set_level(40)  # turn off the warning

    envs = [
        'SawyerPointSingleTask-v0',
        'SawyerPointMultitaskSimple-v0',
        'SawyerPointMultitask-v0',
        'SawyerPickSingleTask-v0',
        'SawyerPickMultitaskSimple-v0',
        'SawyerPickMultitask-v0',
        'SawyerPickReachSingleTask-v0',
        'SawyerPickReachMultitaskSimple-v0',
        'SawyerPickReachMultitask-v0',
    ]

    for name in envs:
        env = gym.make(name)
        obs = env.reset()
        assert obs is not None, f"{name} is tested"
