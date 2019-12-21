import numpy as np
import tensorflow as tf
from kick_ass_utils import compute_kl_two_normals as KL
from bonnie_icm import ICM
import copy
from random import choices

class UnsupervisedBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, emb_dim, time_dim, k_states=None):
        self.buffer_size = buffer_size
        self.time_dim = time_dim
        self.obs_dim = obs_dim
        self.emb_dim = emb_dim
        self.action_dim = action_dim
        self.all_obs = np.zeros((buffer_size, time_dim, obs_dim))
        self.all_actions = np.zeros((buffer_size, time_dim, action_dim))
        self.all_rewards = np.zeros((buffer_size, time_dim, 1))
        self.all_dones = np.zeros((buffer_size, time_dim, 1))
        self.all_zs = np.zeros((buffer_size, emb_dim))
        self.all_probs = np.zeros((buffer_size, time_dim, 1))
        self.obs_for_rl = None
        self.acts_for_rl = None
        self.next_obs_for_rl = None
        self.rewards_for_rl = None
        self.dones_for_rl = None
        self.icm = None
        self.cur_idx = 0
        self.buffer_is_full = False
        self.make_rl_data()
        self.update_rewards = self.update_rewards_v4
        if k_states is not None:
            self.k_states = k_states
        else:
            self.k_states = self.time_dim

    def add_traj(self, traj_states, traj_actions, traj_prob): #, this_z, traj_logps):
        self.all_obs[self.cur_idx, :, :] = traj_states
        self.all_actions[self.cur_idx, :, :] = traj_actions
        self.all_dones[self.cur_idx, -1, 0] = 1
        self.all_probs[self.cur_idx, :] = np.expand_dims(traj_prob, -1)
        #self.all_zs[self.cur_idx, :] = np.squeeze(this_z, 0)

        self.cur_idx += 1
        #self.buffer_slots_filled = self.cur_idx
        if self.cur_idx > self.buffer_size - 1:
            self.cur_idx = 0
            # buffer has filled up at this point.
            self.buffer_is_full = True

    def overwrite_rewards(self, ground_truth_rewards):
        #  don't use this ever. Hack.
        self.all_rewards[self.cur_idx-1, :, 0] = ground_truth_rewards

    def get_training_mb(self, batch_size):
        #  get one state, one action, and one whole trajectory.
        #  used for the encoder BC training.
        if self.buffer_is_full:
            mb_idxs = np.random.randint(self.buffer_size-1, size=batch_size)
        else:
            mb_idxs = np.random.randint(self.cur_idx - 1, size=batch_size)

        mb_whole_obs = self.all_obs[mb_idxs, :, :]
        mb_whole_acts = self.all_actions[mb_idxs, :, :]
        time_idx = np.random.randint(self.time_dim - 1, size=1)
        mb_state = mb_whole_obs[:, time_idx, :]
        mb_act = mb_whole_acts[:, time_idx, :]
        mb_state = mb_state[:, 0, :]  # get rid of the time axis
        mb_act = mb_act[:, 0, :]  # get rid of the time axis.
        mb_prob = self.all_probs[mb_idxs, time_idx]

        return mb_state, mb_act, mb_whole_obs, mb_whole_acts, mb_prob

    def update_rewards_v2(self, encoder):
        #  v2 is supposed to be a faster v1 where you just encode the entire
        #  trajectory at once. Then, you can just write one for loop over time and keep a running
        #  gaussian mean and std.
        #  This is tablbed for now.
        pass

    def update_rewards_v0(self, encoder, batch_size=32):
        all_zs = np.zeros((self.buffer_size, self.time_dim, encoder.encode_dim))
        for t in range(self.time_dim):
            for batch_lb in range(0, self.buffer_size, batch_size):
                obs = self.all_obs[batch_lb:batch_lb+batch_size, 0:t+1, :]
                act = self.all_actions[batch_lb:batch_lb + batch_size, 0:t+1, :]
                #print(obs.shape)
                #print(act.shape)
                zs = encoder.encode(obs, act)
                #print(zs.shape)
                all_zs[batch_lb:batch_lb+batch_size, t, :] = zs
        if self.buffer_is_full:
            max_idx = len(self.all_obs)
        else:
            max_idx = self.cur_idx - 1
        all_zs = all_zs[0:max_idx]
        mean_z = np.mean(all_zs, axis=0)
        mean_z = np.expand_dims(mean_z, 0)
        rew = np.abs(mean_z - all_zs)
        rew = np.sum(rew, axis=2)
        rew = np.expand_dims(rew, -1)
        self.all_rewards[0:max_idx, :, :] = rew
        self.all_rewards = self.all_rewards / np.max(self.all_rewards)
        #self.all_rewards = rew

    #  outline
    def update_rewards_v4(self, encoder, batch_size=32):
        all_rews = np.zeros((self.buffer_size, self.time_dim, 1))
        all_mus = np.zeros((self.buffer_size,  encoder.encode_dim))
        all_vars = np.zeros((self.buffer_size, encoder.encode_dim))
        batch_size = batch_size #min(self.buffer_size, batch_size)
        for batch_lb in range(0, self.buffer_size, batch_size):
            if batch_lb + batch_size < self.buffer_size:
                #print(batch_lb)
                #print(batch_lb + batch_size)
                obs = self.all_obs[batch_lb:batch_lb + batch_size, 0:self.time_dim, :]
                act = self.all_actions[batch_lb:batch_lb + batch_size, 0:self.time_dim, :]
                #print(obs.shape)
                mus, vars = encoder.get_final_mu_and_var(obs[:,-self.k_states:,:], act[:,-self.k_states:,:])
                all_mus[batch_lb:batch_lb + batch_size, :] = mus
                all_vars[batch_lb:batch_lb + batch_size, :] = vars
            #print("blah blah ")
        #total_loss = 0.0
        for traj_idx_i in range(self.buffer_size):
            loss_for_traj_i = 0.0
            for traj_idx_j in range(self.buffer_size):
                loss_for_traj_i += np.mean(KL(all_mus[traj_idx_i], all_vars[traj_idx_i], all_mus[traj_idx_j], all_vars[traj_idx_j]))
            loss_for_traj_i = loss_for_traj_i / self.buffer_size
            #total_loss += loss_for_traj_i
            all_rews[traj_idx_i, -1, :] = loss_for_traj_i
        self.all_rewards = all_rews
        self.all_rewards = self.all_rewards / np.mean(all_rews)
        #self.all_rewards = (self.all_rewards - np.mean(self.all_rewards)) / np.var(self.all_rewards)
        self.all_rewards = self.all_rewards*self.time_dim
        print(np.max(self.all_rewards))
        #print(F"KL reward: {total_loss}")


    def update_rewards_v5(self, encoder, batch_size):
        # ICM

        state_size = self.obs_dim
        action_size = self.action_dim

        if self.icm is None:
            self.icm = ICM(state_size=state_size, action_size=action_size)

        obs = self.all_obs
        a = self.all_actions

        all_rews = np.zeros((self.buffer_size, self.time_dim, 1))
        states, actions, next_states = ([None] * (self.time_dim - 1) for _ in range(3))
        for t in range(self.time_dim - 1):
            rew = self.icm.compute_intrinsic_reward(obs[:, t, :], a[:, t, :], obs[:, t+1, :])
            rew = np.expand_dims(rew, -1)
            # print ("Intrinsic Reward: ", rew)
            all_rews[:, t, :] = rew
            states[t] = obs[:, t, :]
            actions[t] = a[:, t, :]
            next_states[t] = obs[:, t+1, :]
        self.all_rewards = all_rews
        self.all_rewards = self.all_rewards / np.max(self.all_rewards)
        self.all_rewards *= self.time_dim
        # train ICM
        states = np.stack(states).transpose([1, 0, 2]).reshape([-1, state_size])
        next_states = np.stack(next_states).transpose([1, 0, 2]).reshape([-1, state_size])
        actions = np.stack(actions).transpose([1, 0, 2]).reshape([-1, action_size])
        self.icm.batch_train(states, actions, next_states)


    def update_rewards_v6(self, encoder, batch_size):
        self.update_rewards_v0(encoder, batch_size)
        rewz = copy.deepcopy(self.all_rewards)
        self.update_rewards_v5(encoder, batch_size)
        self.all_rewards += rewz


    def update_rewards_v7(self, encoder, batch_size):
        self.update_rewards_v4(encoder, batch_size)
        rewz = copy.deepcopy(self.all_rewards)
        self.update_rewards_v5(encoder, batch_size)
        self.all_rewards += rewz


    def update_rewards_mlp_plus_last_frame_only(self, encoder, batch_size=32):
        all_zs = np.zeros((self.buffer_size, self.time_dim, encoder.encode_dim))
        #t = self.time_dim
        obs = self.all_obs[:, -1, :]
        act = self.all_actions[:, -1, :]
        # print(obs.shape)
        # print(act.shape)
        zs = encoder.encode(obs, act)
        # print(zs.shape)
        all_zs[:, -1, :] = zs
        if self.buffer_is_full:
            max_idx = len(self.all_obs)
        else:
            max_idx = self.cur_idx - 1
        all_zs = all_zs[0:max_idx]
        mean_z = np.mean(all_zs, axis=0)
        mean_z = np.expand_dims(mean_z, 0)
        rew = np.abs(mean_z - all_zs)
        rew = np.sum(rew, axis=2)
        rew = np.expand_dims(rew, -1)
        self.all_rewards[0:max_idx, :, :] = rew
        self.all_rewards = self.all_rewards / np.max(self.all_rewards)


    def make_rl_data(self):
        if self.buffer_is_full:
            max_idx = len(self.all_obs)
        else:
            max_idx = self.cur_idx - 1
        self.obs_for_rl = np.reshape(self.all_obs[0:max_idx, :-1, :], (-1, self.all_obs.shape[2]))
        self.acts_for_rl = np.reshape(self.all_actions[0:max_idx, :-1, :], (-1, self.all_actions.shape[2]))
        self.rewards_for_rl = np.reshape(self.all_rewards[0:max_idx, :-1, :], (-1, self.all_rewards.shape[2]))
        self.next_obs_for_rl = np.reshape(self.all_obs[0:max_idx, 1:, :], (-1, self.all_obs.shape[2]))
        self.dones_for_rl = np.reshape(self.all_dones[0:max_idx, 1:, :], (-1, self.all_dones.shape[2]))

    def sample(self, batch_size=32):
        #  used in SAC and other RL algs.
        #if self.buffer_is_full:
        #    max_idx = len(self.obs_for_rl)
        #else:
        #    #max_idx = (self.cur_idx - 1)*self.time_dim
        #    #max_idx = min()
        max_idx = len(self.obs_for_rl)
        idxs = np.random.choice(max_idx, batch_size, replace=True)
        return self.obs_for_rl[idxs, :], self.acts_for_rl[idxs, :], self.rewards_for_rl[idxs, :], self.next_obs_for_rl[idxs, :], self.dones_for_rl[idxs, :]






class BonnieUnsupervisedBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, time_dim):
        self.buffer_size = buffer_size
        self.time_dim = time_dim
        self.all_obs = np.zeros((buffer_size, time_dim, obs_dim))
        self.all_actions = np.zeros((buffer_size, time_dim, action_dim))
        self.all_rewards = np.zeros((buffer_size, time_dim, 1))
        self.all_dones = np.zeros((buffer_size, time_dim, 1))
        self.obs_for_rl = None
        self.acts_for_rl = None
        self.next_obs_for_rl = None
        self.rewards_for_rl = None
        self.dones_for_rl = None
        self.cur_idx = 0
        self.buffer_is_full = False
        self.make_rl_data()

        self.icm = ICM(state_size = 12, action_size = 2)

    def add_traj(self, traj_states, traj_actions):
        self.all_obs[self.cur_idx, :, :] = traj_states
        self.all_actions[self.cur_idx, :, :] = traj_actions
        self.all_dones[self.cur_idx, -1, 0] = 1
        self.cur_idx += 1
        #self.buffer_slots_filled = self.cur_idx
        if self.cur_idx > self.buffer_size - 1:
            self.cur_idx = 0
            # buffer has filled up at this point.
            self.buffer_is_full = True

    def overwrite_rewards(self, ground_truth_rewards):
        #  don't use this ever. Hack.
        self.all_rewards[self.cur_idx-1, :, 0] = ground_truth_rewards

    def get_training_mb(self, batch_size):
        #  get one state, one action, and one whole trajectory.
        #  used for BCPolWithEncoder
        if self.buffer_is_full:
            mb_idxs = np.random.randint(self.buffer_size-1, size=batch_size)
        else:
            mb_idxs = np.random.randint(self.cur_idx - 1, size=batch_size)
        mb_whole_obs = self.all_obs[mb_idxs, :, :]
        mb_whole_acts = self.all_actions[mb_idxs, :, :]
        time_idxs = np.random.randint(self.time_dim - 1, size=1)
        mb_state = mb_whole_obs[:, time_idxs, :]
        mb_act = mb_whole_acts[:, time_idxs, :]
        mb_state = mb_state[:, 0, :]
        mb_act = mb_act[:, 0, :]
        return mb_state, mb_act, mb_whole_obs, mb_whole_acts


    def update_rewards_v2(self, encoder):
        all_zs = np.zeros((self.buffer_size, self.time_dim, encoder.encode_dim))
        all_mus, all_sigmas = np.zeros((self.buffer_size, self.time_dim, encoder.encode_dim))
        for batch_lb in range(0, self.buffer_size-32, 32):
                obs = self.all_obs[batch_lb:batch_lb+32, :]
                act = self.all_actions[batch_lb:batch_lb + 32, :]
                rrrrr
                #zs = encoder.encode(obs, act)
                mus, sigmas = encoder.get_mus_and_sigmas(obs, act)
                all_mus[batch_lb:batch_lb+32, :, :] = mus
                all_sigmas[batch_lb:batch_lb+32, :, :] = sigmas

        for t in self.time_dim:
            # compute running product
            pass


        all_zs[batch_lb:batch_lb+32] = zs

        mean_z = np.mean(all_zs, axis=0)
        mean_z = np.expand_dims(mean_z, 0)
        rew = np.abs(mean_z - all_zs)
        rew = np.sum(rew, axis=2)
        rew = np.expand_dims(rew, -1)
        self.all_rewards = rew

    def update_rewards(self, encoder, batch_size=32):
        all_zs = np.zeros((self.buffer_size, self.time_dim, encoder.encode_dim))
        for t in range(self.time_dim):
            for batch_lb in range(0, self.buffer_size, batch_size):
                obs = self.all_obs[batch_lb:batch_lb+batch_size, 0:t+1, :]
                act = self.all_actions[batch_lb:batch_lb + batch_size, 0:t+1, :]
                #print(obs.shape)
                #print(act.shape)
                zs = encoder.encode(obs, act)
                #print(zs.shape)
                all_zs[batch_lb:batch_lb+batch_size, t, :] = zs
        if self.buffer_is_full:
            max_idx = len(self.all_obs)
        else:
            max_idx = self.cur_idx - 1
        all_zs = all_zs[0:max_idx]
        mean_z = np.mean(all_zs, axis=0)
        mean_z = np.expand_dims(mean_z, 0)
        rew = np.abs(mean_z - all_zs)
        rew = np.sum(rew, axis=2)
        rew = np.expand_dims(rew, -1)
        self.all_rewards[0:max_idx, :, :] = rew
        #self.all_rewards = rew

    def update_rewards_v5(self):
        # ICM
        state_size = 12
        action_size = 2
        obs = self.all_obs
        a = self.all_actions

        all_rews = np.zeros((self.buffer_size, self.time_dim, 1))
        states, actions, next_states = ([None] * (self.time_dim - 1) for _ in range(3))
        for t in range(self.time_dim - 1):
            rew = self.icm.compute_intrinsic_reward(obs[:, t, :], a[:, t, :], obs[:, t+1, :])
            rew = np.expand_dims(rew, -1)
            # print ("Intrinsic Reward: ", rew)
            all_rews[:, t, :] = rew
            states[t] = obs[:, t, :]
            actions[t] = a[:, t, :]
            next_states[t] = obs[:, t+1, :]
        self.all_rewards = all_rews
        # train ICM
        states = np.stack(states).transpose([1, 0, 2]).reshape([-1, state_size])
        next_states = np.stack(next_states).transpose([1, 0, 2]).reshape([-1, state_size])
        actions = np.stack(actions).transpose([1, 0, 2]).reshape([-1, action_size])
        self.icm.batch_train(states, actions, next_states)

    #  outline
    def update_rewards_v4(self, encoder, batch_size):
        all_rews = []
        for traj_idx_i in range(self.buffer_size):
            loss_for_traj_i = 0.0
            for traj_idx_j in range(self.buffer_size):
                loss_for_traj_i += KL(traj[traj_idx_i], traj[traj_idx_j])
            loss_for_traj_i = loss_for_traj_i / self.buffer_size
            all_rews.append(loss_for_traj_i)


    def make_rl_data(self):
        if self.buffer_is_full:
            max_idx = len(self.all_obs)
        else:
            max_idx = self.cur_idx - 1
        self.obs_for_rl = np.reshape(self.all_obs[0:max_idx, :-1, :], (-1, self.all_obs.shape[2]))
        self.acts_for_rl = np.reshape(self.all_actions[0:max_idx, :-1, :], (-1, self.all_actions.shape[2]))
        self.rewards_for_rl = np.reshape(self.all_rewards[0:max_idx, :-1, :], (-1, self.all_rewards.shape[2]))
        self.next_obs_for_rl = np.reshape(self.all_obs[0:max_idx, 1:, :], (-1, self.all_obs.shape[2]))
        self.dones_for_rl = np.reshape(self.all_dones[0:max_idx, 1:, :], (-1, self.all_dones.shape[2]))


    def sample(self, batch_size=32):
        # used for SAC and RL algs.
        #if self.buffer_is_full:
        #    max_idx = len(self.obs_for_rl)
        #else:
        #    #max_idx = (self.cur_idx - 1)*self.time_dim
        #    #max_idx = min()
        max_idx = len(self.obs_for_rl)
        idxs = np.random.choice(max_idx, batch_size, replace=True)
        return self.obs_for_rl[idxs, :], self.acts_for_rl[idxs, :], self.rewards_for_rl[idxs, :], self.next_obs_for_rl[idxs, :], self.dones_for_rl[idxs, :]




class SimpleBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, time_dim):
        self.buffer_size = buffer_size
        self.time_dim = time_dim
        self.all_obs = np.zeros((buffer_size, time_dim, obs_dim))
        self.all_actions = np.zeros((buffer_size, time_dim, action_dim))
        self.all_rewards = np.zeros((buffer_size, time_dim, 1))
        self.obs_for_rl = None
        self.acts_for_rl = None
        self.next_obs_for_rl = None
        self.rewards_for_rl = None
        self.dones_for_rl = None
        # batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
        self.cur_idx = 0
        self.make_rl_data()
        self.buffer_is_full = False

    def make_rl_data(self):
        self.obs_for_rl = np.reshape(self.all_obs[:, :-1, :], (-1, self.all_obs.shape[2]))
        self.acts_for_rl = np.reshape(self.all_actions[:, :-1, :], (-1, self.all_actions.shape[2]))
        self.rewards_for_rl = np.reshape(self.all_rewards[:, :-1, :], (-1, self.all_rewards.shape[2]))
        self.next_obs_for_rl = np.reshape(self.all_obs[:, 1:, :], (-1, self.all_obs.shape[2]))
        self.dones_for_rl = np.zeros((len(self.next_obs_for_rl), 1))


    def add_traj(self, traj_states, traj_actions):
        self.all_obs[self.cur_idx, :, :] = traj_states
        self.all_actions[self.cur_idx, :, :] = traj_actions
        self.cur_idx += 1
        #self.buffer_slots_filled = self.cur_idx
        if self.cur_idx > self.buffer_size - 1:
            self.cur_idx = 0
            # buffer has filled up at this point.
            self.buffer_is_full = True

    def get_training_mb(self, batch_size):
        #  get one state, one action, and one whole trajectory.
        if self.buffer_is_full:
            mb_idxs = np.random.randint(self.buffer_size-1, size=batch_size)
        else:
            mb_idxs = np.random.randint(self.cur_idx - 1, size=batch_size)

        mb_whole_obs = self.all_obs[mb_idxs, :, :]
        mb_whole_acts = self.all_actions[mb_idxs, :, :]
        time_idxs = np.random.randint(self.time_dim - 1, size=1)
        mb_state = mb_whole_obs[:, time_idxs, :]
        mb_act = mb_whole_acts[:, time_idxs, :]
        mb_state = mb_state[:, 0, :]
        mb_act = mb_act[:, 0, :]
        mb_prob = self.all_probs[mb_idxs, time_idxs]

        return mb_state, mb_act, mb_whole_obs, mb_whole_acts, mb_prob



def test_simple_buffer():
    buf = SimpleBuffer(action_dim=2, obs_dim=12, buffer_size=20, time_dim=30)
    a = np.zeros((30, 2))
    s = np.zeros((30, 12))
    for i in range(100):
        buf.add_traj(s, a)

    for i in range(100):
        mb = buf.get_training_mb(batch_size=32)
        one_ob, one_act, whole_obs, whole_acts = mb



if __name__ == "__main__":
    test_simple_buffer()
