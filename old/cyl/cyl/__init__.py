from gym.envs.registration import register

register(
    id='cyl-v0',
    entry_point='cyl.envs:CylEnv',
)


register(
    id='cyl-sparse-v0',
    entry_point='cyl.envs:CylEnv',
)