import gym
import numpy as np
import tensorflow as tf
from collections import deque
import reinforce as R

def collect_expert_data(env, nsteps, model):
    obs = env.reset()
    ob_list, ac_list, eprew = [], [], 0
    for _ in range(nsteps):
        ob_list.append(obs)
        ac = model.act(obs)
        ac_list.append(ac)
        obs, rew, done, _ = env.step(ac)
        eprew += rew
        if done:
            ob = env.reset()
            print(eprew)
            eprew = 0
    return list(zip(ob_list, ac_list))

class Dagger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        ob_sp = cfg['env'].observation_space
        ac_sp = cfg['env'].action_space
        self.obs = cfg['env'].reset()
        self.st = st = tf.placeholder(tf.float32, [None]+list(ob_sp.shape), name='st')
        self.at = at = tf.placeholder(tf.int32, [None], name='at')

        with tf.variable_scope("dagger"):
            com = R.makeMLP(st, cfg['nn_size'], ac_sp.n, hid_act=tf.nn.tanh, out_act=lambda x: x, no_out=True)
            pi_logits = R.fclayer(com, ac_sp.n, actfn=lambda x: x)
            pd = tf.distributions.Categorical(logits=pi_logits)
            self.sampler = pd.sample()

            self.model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pi_logits,
                              labels=tf.one_hot(at, ac_sp.n, axis=-1)))
            adam = tf.train.AdamOptimizer(learning_rate=cfg['learning_rate'], epsilon=1e-7)
            self._train_pi = adam.minimize(self.model_loss)

        self.data_buf = deque(maxlen=cfg['data_buf_size'])

    def act(self, ob):
        return tf.get_default_session().run([self.sampler], feed_dict={self.st:[ob]})[0][0]

    def act_batch(self, ob_batch):
        return tf.get_default_session().run([self.sampler], feed_dict={self.st:ob_batch})[0]

    def collect_data(self):
        cfg = self.cfg
        env = cfg['env']
        ob_list, ep_rew_list, ep_rew = [], [], 0
        for _ in range(self.cfg['steps_per_epoch']):
            ob_list.append(self.obs.copy())
            ac = self.act(self.obs)
            self.obs, rew, done, _ = env.step(ac)
            ep_rew += rew
            if done:
                ep_rew_list.append(ep_rew)
                ep_rew=0
                self.obs = env.reset()
        expert_ac_list = cfg['expert_model'].act_batch(ob_list)
        return list(zip(ob_list, expert_ac_list)), ep_rew_list

    def train_step(self, S, A):
        _, loss = tf.get_default_session().run([self._train_pi, self.model_loss],
                  feed_dict={self.st:S, self.at:A})
        return loss

    def sample_data(self):
        cfg = self.cfg
        exp_size = len(cfg['expert_data_buf'])
        data_size = len(self.data_buf)
        total_size = exp_size + data_size
        virt_inds = np.random.choice(total_size, cfg['minibatch_size'])
        sample_S, sample_A = [], []
        for idx in virt_inds:
            if idx < exp_size:
                targ_data = cfg['expert_data_buf'][idx]
            else :
                targ_data = self.data_buf[idx-exp_size]
            sample_S.append(targ_data[0])
            sample_A.append(targ_data[1])
        return sample_S, sample_A

    def train_epoch(self):
        cfg = self.cfg
        loss_list = []
        for _ in range(cfg['nminibatches']):
            mb_S, mb_A = self.sample_data()
            loss_list.append(self.train_step(mb_S, mb_A))
        new_data, rew_list = self.collect_data()
        self.data_buf.extend(new_data)
        return np.mean(loss_list), rew_list

    def tf_initialize(self):
        params_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dagger')
        for var in params_model:
            tf.get_default_session().run(var.initializer)

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.__enter__()
    env_id = 'CartPole-v0'
    env = gym.make(env_id)
    cfg = {'env':env, 'gamma':0.99, 'entcoeff':0.01,
           'vf_iters':4, 'vf_batch_size':8,
           'neps_iter':4}
    model = R.REINFORCE_AC(cfg)
    # loadloc = input('Load checkpoint name:')
    loadloc = 'test'
    model.load('models/'+loadloc+'/model.ckpt')
    env = gym.make(env_id)
    cfg = {'env':env, 'nn_size':[4,4], 'learning_rate':3e-4,
           'data_buf_size':25000, 'steps_per_epoch':256,
           'expert_model':model, 'minibatch_size':32,
           'nminibatches':8, 'expert_data_buf': None}
    dgr = Dagger(cfg)
    dgr.tf_initialize()
    cfg['expert_data_buf'] = collect_expert_data(env, 2500, model)

    eprewbuf = deque(maxlen=10)
    for epoch in range(1000):
        loss, rews = dgr.train_epoch()
        eprewbuf.extend(rews)
        if (epoch+1) % 10 == 0 :
            print('Epoch', epoch)
            print('Loss: ',loss, 'Reward:', np.mean(eprewbuf))
            print('=============')
