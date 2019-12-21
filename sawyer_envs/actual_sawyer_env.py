from sawyer_envs.pick_place import pick_place_env
import numpy as np



def get_sawyer_reach_env():
    #  sawyer reach.
    kwargs = dict(frame_skip=5, reward_type="reach_dense",
                  mocap_low=(-0.05, 0.35, 0.05),
                  mocap_high=(0.45, 0.7, 0.35),
                  hand_low=(-0.05, 0.35, 0.05),
                  hand_high=(0.45, 0.7, 0.35),
                  obj_low=(0.05, 0.45, 0.02),
                  obj_high=(0.35, 0.6, 0.02)
                  )

    env = pick_place_env(**kwargs)
    return env



def get_other_sawyer_envs(env_name='reach'):
        if env_name == 'reach':
            kwargs=dict(frame_skip=5, reward_type="reach_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
        elif env_name == 'hover':
            kwargs=dict(frame_skip=5, reward_type="hover_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
        elif env_name == 'hover_no_touch':
            kwargs=dict(frame_skip=5, reward_type="hover_dense",
                        # note: No way that this touches the block
                        mocap_low=(-0.05, 0.35, 0.1),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.1),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
        elif env_name == 'hover_hand_rotated':
            kwargs= dict(frame_skip=5, reward_type="hover_dense",
                        effector_quat=(0.5, -0.5, 0.5, 0.5,),
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        elif env_name == 'touch':
            kwargs=dict(frame_skip=5, reward_type="touch_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )
        elif env_name == 'push':
            kwargs=dict(frame_skip=5, reward_type="push_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        elif env_name == 'pick':
            kwargs=dict(frame_skip=5, reward_type="pick_dense",
                        mocap_low=(-0.05, 0.35, 0.035),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        elif env_name == 'pick_reach':
            kwargs=dict(frame_skip=5, reward_type="pick_reach_dense",
                        mocap_low=(-0.05, 0.35, 0.035),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.1),
                        obj_high=(0.35, 0.6, 0.30),
                        shaped_init=0.5,
                        )

        elif env_name == 'pick_place':
            kwargs = dict(frame_skip=5, reward_type="pick_place_dense",
                        mocap_low=(-0.05, 0.35, 0.05),
                        mocap_high=(0.45, 0.7, 0.35),
                        hand_low=(-0.05, 0.35, 0.05),
                        hand_high=(0.45, 0.7, 0.35),
                        obj_low=(0.05, 0.45, 0.02),
                        obj_high=(0.35, 0.6, 0.02)
                        )

        env = pick_place_env(**kwargs)
        return env

def main():
    env = get_sawyer_env()
    env.reset()
    for i in range(100000):
        env.step(np.random.randn(4))
        env.render()


if __name__ == "__main__":
    main()
