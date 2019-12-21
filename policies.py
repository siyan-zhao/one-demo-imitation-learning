import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as L
import copy
from envs.cyl import CylEnv, DummyCylEnv
from baselines.common.tf_util import function
from kick_ass_utils import euclidean_loss, MSE, eval_class_acc, init_tf
from sac import SAC
from stable_baselines.sac.policies import MlpPolicy


class GaussianTrajectorEncoder:
    def __init__(self, time_dim, obs_dim, action_dim, batch_size=32, encode_dim=16):
        self.encode_dim = encode_dim
        self.net_size = 32
        self.horizon = time_dim

        #self.num_of_tasks_in_batch = 5
        #self.num_of_states_pre_task_in_batch = 32

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.latent_size = self.encode_dim

        #self.batch_size = batch_size #tf.placeholder(tf.float32, [])
        #self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        #self.num_of_states_pre_task_in_batch = tf.placeholder(tf.float32, [])
        self.batch_size = batch_size
        self.whole_traj = tf.placeholder(tf.float32, [self.batch_size, self.horizon,
                                                      self.obs_dim])
        self.whole_act = tf.placeholder(tf.float32, [self.batch_size, self.horizon,
                                                     self.action_dim])
        self.context = tf.concat([self.whole_traj, self.whole_act], axis=-1)

        self.tf_infer_posterior_v2()

    def tf_context_encoder(self, context):
        """ context encoder MLP network; 
        input: a single context[state i, action i], 
        output = [num_traj, 2 * latent_size] """
        with tf.variable_scope("context_encoder", reuse=False):
            emb = tf.layers.dense(context, self.net_size, activation=tf.nn.relu, use_bias=True, name='mlp_layer1')
            emb = tf.layers.dense(emb, self.net_size, activation=tf.nn.relu, use_bias=True, name='mlp_layer3')
            out = tf.layers.dense(emb, self.latent_size * 2, activation=None, use_bias=True)
        """ output dimension is [laten_size + latent_size], one for mu, one for sigma. As in PEARL ) """
        return out

    def tf_product_of_gaussians(self, mus, sigmas_squared):
        """ compute mu, sigma of product of gaussians """
        sigmas_squared = tf.clip_by_value(sigmas_squared, clip_value_min=1e-7,
                                          clip_value_max=1e+7)  # to avoid negative sigmas
        #tf.math_ops.reciprocal
        out_sigma_squared = 1. / tf.reduce_sum(tf.math.reciprocal(sigmas_squared), 0)
        mu = out_sigma_squared * tf.reduce_sum(mus / sigmas_squared, 0)

        return mu, out_sigma_squared

    def tf_sample_z(self):
        """ sample by a reparameterize trick """

        #epsilon = tf.random_normal(shape=self.z_means.shape)
        #self.z = epsilon * tf.math.exp(self.z_vars * 0.5) + self.z_means
        self.z = tf.random_normal(shape=self.z_means.shape, mean=self.z_means, stddev=self.z_vars)
        #random_normal is actually reparameterized.

    def tf_gaussians_no_product(self, mus, sigmas_squared):
        sigmas_squared = tf.clip_by_value(sigmas_squared, clip_value_min=1e-7,
                                          clip_value_max=1e+7)  # to avoid negative sigmas
        # tf.math_ops.reciprocal
        out_sigma_squared = 1. / tf.math.reciprocal(sigmas_squared)
        mu = out_sigma_squared * (mus / tf.math.reciprocal(sigmas_squared))


    def tf_infer_posterior(self):
        """ compute q(z|c) as a function of input context and sample new z from it """
        ''' assume context shape is [num_traj, num_of_time_step, num_context_encoder_output]'''
        # print("---context---", self.context)
        params = self.context_encoder(self.context)  # context: [num_task, num_time_step, action + observation]
        params = tf.reshape(params,
                            [self.batch_task_size, self.num_time_step_per_traj, self.context_encoder_output_size])
        print("params", params)
        mus = params[..., : self.latent_size]
        sigmas_squared = tf.math.softplus(params[..., self.latent_size:])

        z_params = [self.product_of_gaussians(m, s) for m, s in zip(tf.unstack(mus), tf.unstack(sigmas_squared))]

        self.z_means = tf.reshape(tf.stack([x[0] for x in z_params]), [self.batch_task_size, self.latent_size])
        self.z_vars = tf.reshape(tf.stack([x[1] for x in z_params]), [self.batch_task_size, self.latent_size])

        # print(self.z_means) #shape=(num_batch_tasks, latent_Dim )
        # print(self.z_vars)  #shape=(num_batch_tasks, latent_Dim )
        self.sample_z()


    def tf_infer_posterior_v2(self):
        """ compute q(z|c) as a function of input context and sample new z from it """
        ''' assume context shape is [num_traj, num_of_time_step, num_context_encoder_output]'''
        # print("---context---", self.context)
        params = self.tf_context_encoder(self.context)  # context: [num_task, num_time_step, action + observation]

        #params = tf.reshape(params, [self.num_tasks_in_batch, self.num_of_states_pre_task_in_batch,
        #                             self.latent_size*2])
        print("params", params)
        mus = params[..., : self.latent_size]
        sigmas_squared = tf.nn.softplus(params[..., self.latent_size:])
        self.mus = mus
        self.sigmas_squared = sigmas_squared
        self.get_mus_and_sigmas = function(inputs=[self.whole_traj, self.whole_act], outputs=[self.mus, self.sigmas_squared])
        z_params = [self.tf_product_of_gaussians(m, s) for m, s in zip(tf.unstack(mus), tf.unstack(sigmas_squared))]

        self.z_means = tf.reshape(tf.stack([x[0] for x in z_params]), [self.batch_size, self.latent_size])

        self.z_vars = tf.reshape(tf.stack([x[1] for x in z_params]), [self.batch_size, self.latent_size])
        # print(self.z_means) #shape=(num_batch_tasks, latent_Dim )
        # print(self.z_vars)  #shape=(num_batch_tasks, latent_Dim )
        self.tf_sample_z()
        self.encode_func = function(inputs=[self.whole_traj, self.whole_act], outputs=[self.z])
        self.get_final_mu_and_var = function(inputs=[self.whole_traj, self.whole_act], outputs=[self.z_means, self.z_vars])
        return self.z


    def encode(self, s, a):
        s_shape = s.shape
        if len(s_shape) == 2:
            s = np.repeat(s[np.newaxis, ...], self.batch_size, axis=0)
            a = np.repeat(a[np.newaxis, ...], self.batch_size, axis=0)
            #s = np.tile(s, self.batch_size)
            #a = np.tile(a, self.batch_size)
            z_out = self.encode_func(s, a)[0]
            #sess = tf.get_default_session()
            #fd = {self.whole_traj: s, self.whole_act: a}
            #z_np = sess.run(self.z, feed_dict=fd)
            return z_out[0]
        else:
            z_out = self.encode_func(s, a)[0]
            return z_out



def test_gaussian_traj_encoder():
    sess = tf.Session()
    with sess.as_default():
        action_dim = 6
        obs_dim = 12
        batch_size = 32
        time_steps = 50
        encoder = GaussianTrajectorEncoder(action_dim=6, obs_dim=12, batch_size=batch_size)
        kkkkk
        #z_tf = encoder.tf_infer_posterior_v2()
        z_tf = encoder.z
        whole_traj, whole_act = encoder.whole_traj, encoder.whole_act
        #s_encoder = np.zeros((batch_size, time_steps, obs_dim))
        #a_encoder = np.zeros((batch_size, time_steps, action_dim))
        #z_encoded_np = encoder.encode(s_encoder[0], a_encoder[0])
        #z_encoded_np_2 = encoder.encode(s_encoder, a_encoder)

        s_mb, a_mb = np.zeros((batch_size, obs_dim)), np.zeros((batch_size, action_dim))

        bc_pol = BCPolWithEncoder(s_mb, a_mb, gaussian_encoder=encoder, discrete=False)

        init_tf()

        s_encoder = np.zeros((batch_size, time_steps, obs_dim))
        a_encoder = np.zeros((batch_size, time_steps, action_dim))
        action = bc_pol.act(s_mb[0], s_encoder, a_encoder)

        for i in range(10):
            bc_pol.train(s_mb, a_mb, s_encoder, a_encoder)
            print(F"Finished training step {i}")



class BCPol:
    def __init__(self, s_mb, a_mb, discrete=True):
        self.s_shp = s_mb.shape
        self.a_shp = a_mb.shape
        self.discrete = discrete
        self.nn_input_s = tf.placeholder(dtype=tf.float32, shape=[None, self.s_shp[1]])
        self.nn_input_a = tf.placeholder(dtype=tf.float32, shape=[None, self.a_shp[1]])
        self.make_model()

    def make_model(self):
        h = L.fully_connected(self.nn_input_s, 32)
        h = L.fully_connected(h, 32)
        self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)
        if self.discrete is True:
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.nn_input_a, logits=self.out)
            self.out = tf.nn.softmax(self.out)
        if self.discrete is False:
            loss = euclidean_loss(y_true=self.nn_input_a, y_pred=self.out)

        update = tf.train.AdamOptimizer().minimize(loss)

        self.output_fn = function(inputs=[self.nn_input_s], outputs=[self.out])
        self.train_fn = function(inputs=[self.nn_input_s, self.nn_input_a], outputs=[loss], updates=[update])

    def train(self, s_mb, a_mb):
        loss = self.train_fn(s_mb, a_mb)
        return np.mean(loss[0])

    def eval(self, s_mb, a_mb):
        if self.discrete is False:
            a_hat = self.act(s_mb)
            return MSE(a_hat, a_mb)
        else:
            a_hat = self.act(s_mb)
            return eval_class_acc(a_hat, a_mb)

    def act(self, state):
        state = np.reshape(state,[-1,28])
        a = self.output_fn(state)
        a = a[0]
        if self.discrete is True:
            # a = np.argmax(a, axis=1)
            return a
        else:
            return a


class BCPolWithEncoder:
    def __init__(self, s_mb, a_mb, z_mb, gaussian_encoder, discrete=False, encoder_type='concat'):
        # encoder types: concat, net, cbn.
        # concat just concats to input. net runs z through its own neural net then concats.
        # cbn uses conditional batch normalization.
        self.gaussian_encoder = gaussian_encoder
        self.encoder_type = encoder_type
        self.s_shp = s_mb.shape
        self.a_shp = a_mb.shape
        self.z_shp = z_mb.shape
        self.latent_size = 16
        self.bate = 1
        self.discrete = discrete
        self.nn_input_s = tf.placeholder(dtype=tf.float32, shape=[self.s_shp[0], self.s_shp[1]])
        self.nn_input_a = tf.placeholder(dtype=tf.float32, shape=(self.a_shp[0], self.a_shp[1]))
        self.nn_input_z = tf.placeholder(dtype=tf.float32, shape=(None, self.z_shp[1]))
        self.nn_input_p = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.nn_input_z = gaussian_encoder.z   #.tf_infer_posterior_v2()
        self.whole_traj, self.whole_act = self.gaussian_encoder.whole_traj, self.gaussian_encoder.whole_act
        self.means, self.vars = self.gaussian_encoder.z_means, self.gaussian_encoder.z_vars
        self.make_model()

    def gaussian_likelihood(self, x, mu, log_std):
        EPS = 1e-8
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def imitation_network_as_explorer(self, z, state):
        sess = tf.get_default_session()
        z = np.repeat(z, 10, axis=0)
        state = np.expand_dims(state, 0)
        state = np.repeat(state, 10, axis=0)
        #sess.run(self.nn_input_z.assign[z])
        a = sess.run(self.sample_action, feed_dict={self.nn_input_s: state, self.nn_input_z: z})
        p = sess.run(self.logp, feed_dict={self.nn_input_s: state, self.nn_input_z: z})
        #print(a,'action')
        #print(p, 'prob')
        mu = 0
        return a[0], p[0], mu

    def make_model(self):
        if self.encoder_type == "concat":
            nn_input = tf.concat([self.nn_input_s, self.nn_input_z], axis=1)
            h = L.fully_connected(nn_input, 32)
            h = L.fully_connected(h, 32)
            self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)
        elif self.encoder_type == "cbn":
            nn_input = self.nn_input_s
            h = L.fully_connected(nn_input, 32)
            h = L.fully_connected(h, 32)
            self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)
        else:  # net encoding
            nn_input = self.nn_input_s
            h = L.fully_connected(nn_input, 32)
            h = L.fully_connected(h, 32)
            h_enc = L.fully_connected(self.nn_input_z, 32)  # z gets it's own encoding
            h_enc = L.fully_connected(h_enc, 16)
            h_plus_h_enc = tf.concat([h_enc, h], axis=1)  # concat with state network
            h = L.fully_connected(h_plus_h_enc, 12)
            self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)

        # get mean and std
        self.mus = L.fully_connected(inputs=self.out, num_outputs=self.a_shp[1], activation_fn=tf.nn.tanh)
        self.log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(self.a_shp[1], dtype=np.float32))
        self.std = tf.exp(self.log_std)
        # sample action
        self.sample_action = self.mus + tf.random_normal(tf.shape(self.mus)) * self.std

        self.logp = self.gaussian_likelihood(self.sample_action, self.mus, self.log_std)
        self.logp_pi = self.gaussian_likelihood(self.nn_input_a, self.mus, self.log_std)

        # uncomment it to output deterministic action:
        #self.sample_action = self.out

        if self.discrete is True:
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.nn_input_a, logits=self.out)
            self.out = tf.nn.softmax(self.out)
        if self.discrete is False:
            self.loss = euclidean_loss(y_true=self.nn_input_a, y_pred=self.sample_action)

        #print(self.logp_pi.eval(),'?????')
        # uncomment to use gaussian likelihood loss
        self.loss = tf.exp(self.logp_pi - self.nn_input_p)
        update = tf.train.AdamOptimizer().minimize(self.loss)
        #update = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.95).minimize(loss)

        self.output_fn = function(inputs=[self.nn_input_s, self.whole_traj, self.whole_act], outputs=[self.sample_action, self.logp])
        self.train_fn = function(inputs=[self.nn_input_s, self.whole_traj, self.whole_act, self.nn_input_a, self.nn_input_p],
                                 outputs=[self.loss], updates=[update])
        self.compute_z_fn = function(inputs=[self.whole_traj, self.whole_act],
                                     outputs=[self.nn_input_z, self.means, self.vars])

    def train(self, s_mb, a_mb, p_mb, whole_traj_mb, whole_act_mb):
        loss = self.train_fn(s_mb, whole_traj_mb, whole_act_mb, a_mb, p_mb)
        return np.mean(loss[0])

    def compute_z(self, whole_traj, whole_act):
        z_hat, _, _ = self.compute_z_fn(whole_traj, whole_act)
        return z_hat

    def eval(self, s_mb, whole_traj_mb, whole_act_mb, a_mb):
        if self.discrete is False:
            a_hat, _ = self.act(s_mb, whole_traj_mb, whole_act_mb)
            return MSE(a_hat, a_mb)
        else:
            a_hat = self.act(s_mb, whole_traj_mb, whole_act_mb)
            return eval_class_acc(a_hat, a_mb)

    def act(self, state, whole_traj_mb, whole_act_mb):
        s_shape = state.shape
        if len(s_shape) == 1:

            state = np.repeat(state[np.newaxis, ...], self.s_shp[0], axis=0)
            whole_traj_mb = np.repeat(whole_traj_mb[np.newaxis, ...], self.s_shp[0], axis=0)
            whole_act_mb = np.repeat(whole_act_mb[np.newaxis, ...], self.s_shp[0], axis=0)
            a, p = self.output_fn(state, whole_traj_mb, whole_act_mb)
            a = a[0]
            return a, p
        else:
            #a = self.output_fn(state, whole_traj_mb, whole_act_mb)[0]
            sess = tf.get_default_session()
            feed_dict = {self.whole_traj: whole_traj_mb, self.whole_act: whole_act_mb,
                         self.nn_input_s: state}
            a = sess.run(self.out, feed_dict=feed_dict)
            p = 0
            return a, p


class SACPol:
    def __init__(self, env, total_training_steps, explore_steps=100):
        self.env = env
        pol = MlpPolicy
        self.sac = SAC(policy=pol, env=env)
        self.total_training_steps = total_training_steps
        self.training_steps_so_far = 0
        self.explore_steps = explore_steps

    def act(self, state, deterministic=False):
        a = self.sac.policy_tf.step(state[None], deterministic=deterministic).flatten()
        #print(a.shape)
        rescaled_action = a * np.abs(self.env.action_space.low)
        return a, rescaled_action
        #a, _ = self.sac.predict(state, deterministic=False)
        #return a

    #def act_eval(self, state):
    #    a, _ = self.sac.predict(state, deterministic=True)
    #    return a

    def train(self, replay_buffer, num_timesteps):
        self.training_steps_so_far += num_timesteps
        self.sac.train_step(replay_buffer, num_timesteps,
                            total_training_steps_done_so_far=self.training_steps_so_far,
                            total_training_timesteps=self.total_training_steps)


def test_sac_pol():
    from kick_ass_alg_v2 import rollout
    from buffers import UnsupervisedBuffer
    sess = tf.Session()
    with sess.as_default():
        env = DummyCylEnv()
        random_pol = RandomPol(env=env)
        buffer = UnsupervisedBuffer(buffer_size=32 * 16, obs_dim=env.obs_dim, action_dim=env.action_dim,
                                    time_dim=200)
        for i in range(32*16):
            traj_states, traj_actions = rollout(random_pol, env, horizon=200)
            buffer.add_traj(traj_states, traj_actions)
        buffer.make_rl_data()
        sac = SACPol(env, total_training_steps=10000)

        state = np.zeros((env.obs_dim,))
        action = sac.act(state=state)
        sac.train(buffer, num_timesteps=10)



class RandomPol:
    def __init__(self, env):
        self.env = env

    def train(self, replay_buffer, num_timesteps):
        pass

    def act(self, state):
        a = np.random.randn(self.env.action_dim)
        return a, a


class OneShotImitationPol:
    #  from the paper  'one shot imitation learning'
    def __init__(self, s_mb, a_mb):
        self.s_shp = s_mb.shape
        self.a_shp = a_mb.shape
        self.batch_size = self.s_shp[0]
        self.obs_dim = self.s_shp[1]
        self.action_dim = self.a_shp[1]
        self.nn_input_s = tf.placeholder(dtype=tf.float32, shape=[self.s_shp[0], self.s_shp[1]])
        self.nn_input_a = tf.placeholder(dtype=tf.float32, shape=(self.a_shp[0], self.a_shp[1]))
        self.whole_traj = tf.placeholder(tf.float32, [self.batch_size, None,
                                                      self.obs_dim])
        self.whole_act = tf.placeholder(tf.float32, [self.batch_size, None,
                                                     self.action_dim])
        self.context = tf.concat([self.whole_traj, self.whole_act], axis=-1)
        self.c_in = None
        self.rnn_hidden_state = None
        self.out = None
        self.rnn_out = None
        self.make_rnn()
        self.make_model()

    def make_rnn(self):
        nhid = 32
        rnn_cell = tf.nn.rnn_cell.GRUCell(nhid)
        # initial_state = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        # initial_state = tf.placeholder(tf.float32, batch_size, state_size

        self.c_in = tf.placeholder(tf.float32,
                                   shape=[None, rnn_cell.state_size],
                                   name='c_in')
        # h_in = tf.placeholder(tf.float32,
        #                      shape=[1, rnn_cell.state_size.h],
        #                      name='h_in')
        # state_in = [c_in, h_in]

        state_in = self.c_in  # rnn.LSTMStateTuple(c_in, h_in)

        outputs, self.rnn_hidden_state = tf.nn.dynamic_rnn(rnn_cell, self.context,
                                                           # sequence_length=self.seq_len,
                                                           initial_state=state_in,
                                                           dtype=tf.float32,
                                                           time_major=False)

        h0 = outputs  # [:, :, :]  # get last output.
        #h0 = L.fully_connected(h0, nhid)
        h0 = h0[:, -1, :]
        self.rnn_out = h0
        self.rnn_out = L.fully_connected(h0, nhid)
        #self.Y_hat = L.fully_connected(h0, YS.shape[2], activation_fn=tf.identity)

    def make_model(self):
        nn_input = tf.concat([self.nn_input_s, self.rnn_out], axis=1)
        h = L.fully_connected(nn_input, 32)
        h = L.fully_connected(h, 32)
        self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)


        loss = euclidean_loss(y_true=self.nn_input_a, y_pred=self.out)

        update = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
        #update = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.95).minimize(loss)

        self.output_fn = function(inputs=[self.nn_input_s, self.whole_traj, self.whole_act, self.c_in], outputs=[self.out])
        self.train_fn = function(inputs=[self.nn_input_s, self.whole_traj, self.whole_act, self.nn_input_a, self.c_in],
                                 outputs=[loss], updates=[update])

    def train(self, s_mb, a_mb, whole_traj_mb, whole_act_mb):
        c_in_np = np.zeros((self.batch_size, 32))
        loss = self.train_fn(s_mb, whole_traj_mb, whole_act_mb, a_mb, c_in_np)
        return np.mean(loss[0])

    def eval(self, s_mb, whole_traj_mb, whole_act_mb, a_mb):
        a_hat = self.act(s_mb, whole_traj_mb, whole_act_mb)
        return MSE(a_hat, a_mb)


    def act(self, state, whole_traj_mb, whole_act_mb):
        c_in_np = np.zeros((self.batch_size, 32))
        s_shape = state.shape
        if len(s_shape) == 1:
            state = np.repeat(state[np.newaxis, ...], self.s_shp[0], axis=0)
            whole_traj_mb = np.repeat(whole_traj_mb[np.newaxis, ...], self.s_shp[0], axis=0)
            whole_act_mb = np.repeat(whole_act_mb[np.newaxis, ...], self.s_shp[0], axis=0)
            a = self.output_fn(state, whole_traj_mb, whole_act_mb, c_in_np)[0]
            a = a[0]
            return a
        else:
            #a = self.output_fn(state, whole_traj_mb, whole_act_mb)[0]
            sess = tf.get_default_session()
            feed_dict = {self.whole_traj: whole_traj_mb, self.whole_act: whole_act_mb,
                         self.nn_input_s: state, self.c_in: c_in_np}
            a = sess.run(self.out, feed_dict=feed_dict)
            return a




class MLPPolWithLastStateEncoder:
    def __init__(self, s_mb, a_mb, discrete=False, encoder_type='concat'):
        # encoder types: concat, net, cbn.
        # concat just concats to input. net runs z through its own neural net then concats.
        # cbn uses conditional batch normalization.
        self.encoder_type = encoder_type
        self.s_shp = s_mb.shape
        self.a_shp = a_mb.shape
        #self.z_shp = z_mb.shape
        self.discrete = discrete
        self.nn_input_s = tf.placeholder(dtype=tf.float32, shape=[None, self.s_shp[1]])
        self.nn_input_a = tf.placeholder(dtype=tf.float32, shape=(None, self.a_shp[1]))
        #self.nn_input_z = tf.placeholder(dtype=tf.float32, shape=(None, self.z_shp[1]))
        self.nn_input_last_s = tf.placeholder(dtype=tf.float32, shape=[None, self.s_shp[1]])
        self.nn_input_last_a = tf.placeholder(dtype=tf.float32, shape=(None, self.a_shp[1]))
        self.context = tf.concat([self.nn_input_last_s, self.nn_input_last_a], axis=-1)
        self.encoded_h = None
        self.encode_dim = 16
        self.make_model()

    def make_model(self):
        if self.encoder_type == "concat":
            enc = L.fully_connected(self.context, self.encode_dim)
            self.encoded_h = enc
            nn_input = tf.concat([self.nn_input_s, self.encoded_h], axis=1)
            h = L.fully_connected(nn_input, 32)
            h = L.fully_connected(h, 32)
            self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)
        elif self.encoder_type == "cbn":
            nn_input = self.nn_input_s
            h = L.fully_connected(nn_input, 32)
            h = L.fully_connected(h, 32)
            self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)
        else:  # net encoding
            nn_input = self.nn_input_s
            h = L.fully_connected(nn_input, 32)
            h = L.fully_connected(h, 32)
            h_enc = L.fully_connected(self.nn_input_z, 32)  # z gets it's own encoding
            h_enc = L.fully_connected(h_enc, 16)
            h_plus_h_enc = tf.concat([h_enc, h], axis=1)  # concat with state network
            h = L.fully_connected(h_plus_h_enc, 12)
            self.out = L.fully_connected(h, self.a_shp[1], activation_fn=tf.identity)



        loss = euclidean_loss(y_true=self.nn_input_a, y_pred=self.out)

        update = tf.train.AdamOptimizer().minimize(loss)
        #update = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.95).minimize(loss)

        self.output_fn = function(inputs=[self.nn_input_s, self.nn_input_last_s, self.nn_input_last_a], outputs=[self.out])
        self.train_fn = function(inputs=[self.nn_input_s, self.nn_input_last_s, self.nn_input_last_a, self.nn_input_a],
                                 outputs=[loss], updates=[update])
        self.encode_fn = function(inputs=[self.nn_input_last_s, self.nn_input_last_a], outputs=[self.encoded_h])

    def train(self, s_mb, a_mb, whole_traj_mb, whole_act_mb):
        loss = self.train_fn(s_mb, whole_traj_mb[:, -1, :], whole_act_mb[:, -1, :], a_mb)
        return np.mean(loss[0])

    def eval(self, s_mb, whole_traj_mb, whole_act_mb, a_mb):
        if self.discrete is False:
            a_hat = self.act(s_mb, whole_traj_mb, whole_act_mb)
            return MSE(a_hat, a_mb)
        else:
            a_hat = self.act(s_mb, whole_traj_mb, whole_act_mb)
            return eval_class_acc(a_hat, a_mb)

    def act(self, state, whole_traj_mb, whole_act_mb):
        s_shape = state.shape
        if len(s_shape) == 1:
            state = np.repeat(state[np.newaxis, ...], self.s_shp[0], axis=0)
            whole_traj_mb = np.repeat(whole_traj_mb[np.newaxis, ...], self.s_shp[0], axis=0)
            whole_act_mb = np.repeat(whole_act_mb[np.newaxis, ...], self.s_shp[0], axis=0)
            a = self.output_fn(state, whole_traj_mb[:, -1, :], whole_act_mb[:, -1, :])[0]
            a = a[0]
            return a
        else:
            #a = self.output_fn(state, whole_traj_mb, whole_act_mb)[0]
            sess = tf.get_default_session()
            feed_dict = {self.nn_input_last_s: whole_traj_mb[:, -1, :], self.nn_input_last_a: whole_act_mb[:, -1, :],
                         self.nn_input_s: state}
            a = sess.run(self.out, feed_dict=feed_dict)
            return a

    def encode(self, s, a):
        #print(s.shape)
        #print(a.shape)
        sess = tf.get_default_session()
        feed_dict = {self.nn_input_last_s: s, self.nn_input_last_a: a}
        h = sess.run(self.encoded_h, feed_dict=feed_dict)
        #print(h.shape)
        return h
        #return self.encode_fn(s, a)[0]




def test_one_shot_imitation_pol():
    sess = tf.Session()
    with sess.as_default():
        s = np.zeros((32, 9))
        a = np.zeros((32, 4))
        pol = OneShotImitationPol(a_mb=a, s_mb=s)
        init_tf()
        whole_s = np.zeros((32, 14, 9))
        whole_a = np.zeros((32, 14, 4))
        a = pol.act(state=s, whole_traj_mb=whole_s, whole_act_mb=whole_a)
        pol.train(s_mb=s, a_mb=a, whole_traj_mb=whole_s, whole_act_mb=whole_a)



def test_bc_pol():
    sess = tf.Session()
    with sess.as_default():
        s = np.zeros((32, 9))
        a = np.zeros((32, 4))
        pol = BCPol(s_mb=s, a_mb=a, discrete=False)
        init_tf()
        a = pol.act(state=s)
        pol.train(s_mb=s, a_mb=a)


def test_bc_pol_with_encoder():
    sess = tf.Session()
    with sess.as_default():
        s = np.zeros((32, 9))
        a = np.zeros((32, 4))
        z = np.zeros((32, 2))
        pol = BCPolWithEncoder(s_mb=s, a_mb=a, z_mb=z, discrete=False)
        init_tf()
        a = pol.act(state=s, encoded_z=z)
        print(F"Action: {a[0]}")
        loss = pol.train(s_mb=s, a_mb=a, z_mb=z)
        print(F"Loss is: {loss}")


def main():
    test_one_shot_imitation_pol()


if __name__ == "__main__":
    main()
