import gym
import numpy as np
import tensorflow as tf
from collections import deque

def fclayer(x, nunits, actfn=tf.nn.relu):
    W = tf.get_variable("weights", shape=[x.shape[1], nunits], dtype=tf.float32,
        initializer=tf.random_normal_initializer())
    b = tf.get_variable("biases", shape=[nunits], dtype=tf.float32,
        initializer=tf.random_normal_initializer())
    return actfn(tf.matmul(x,W) + b)

def makeMLP(X, hidden_sizes, out_size, reuse=None, hid_act=tf.nn.relu, out_act=lambda x: x, no_out=False):
    x = X
    for i,nh in enumerate(hidden_sizes) :
        with tf.variable_scope('fc%d'%(i+1), reuse=reuse):
            x = fclayer(x, nh, actfn=hid_act)
    if not no_out:
        with tf.variable_scope('out', reuse=reuse):
            o = fclayer(x, out_size, actfn=out_act)
            return o
    else:
        return x

class REINFORCE_AC:
    def __init__(self, cfg):
        self.cfg = cfg
        ob_sp = cfg['env'].observation_space
        ac_sp = cfg['env'].action_space
        self.st = st = tf.placeholder(tf.float32, [None]+list(ob_sp.shape), name='st')
        self.at = at = tf.placeholder(tf.int32, [None], name='at')
        self.At = At = tf.placeholder(tf.float32, [None], name='At')
        self.Vtarg = Vtarg = tf.placeholder(tf.float32, [None], name='Vtarg')
        with tf.variable_scope("policy"):
            com = makeMLP(st, [4,4], ac_sp.n, hid_act=tf.nn.tanh, out_act=lambda x: x, no_out=True)
            pi_logits = fclayer(com, ac_sp.n, actfn=lambda x: x)
            # pi_logits = tf.clip_by_value(pi_logits, clip_value_min=-5, clip_value_max=5)
        pd = tf.distributions.Categorical(logits=pi_logits)
        self.logpi = pd.log_prob(at)
        self.sampler = pd.sample()
        self.entropy = pd.entropy()
        # Critic
        with tf.variable_scope("critic"):
            com = makeMLP(st, [4,4], ac_sp.n, hid_act=tf.nn.relu, out_act=lambda x: x, no_out=True)
            critic_out = fclayer(com, 1, actfn=lambda x: x)
            # critic_out = tf.clip_by_value(critic_out, clip_value_min=-5, clip_value_max=5)
        self.vf = tf.reshape(critic_out, [-1])
        self.vf_loss = tf.reduce_mean(tf.square(Vtarg-self.vf))
        self.policyloss = -tf.reduce_mean((cfg['entcoeff']*self.entropy + self.logpi*At))
        with tf.variable_scope("policy"):
            params_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy')
        # with tf.variable_scope("critic"):
        #     params_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        self.policylossgrad = policylossgrad = tf.gradients(self.policyloss, params_model)
        # self.vflossgrad = vflossgrad = tf.gradients(self.vf_loss, params_critic)
        self.pg_ph = [tf.placeholder(tf.float32, name='pg_%d_grad'%(idx)) for idx,var in enumerate(params_model)]
        # self.vfg_ph = [tf.placeholder(tf.float32, name='vf_%d_grad'%(idx)) for idx,var in enumerate(params_critic)]
        adam = tf.train.AdamOptimizer(learning_rate=5e-3, epsilon=1e-7)
        self._train_pi_grads = adam.apply_gradients(list(zip(self.pg_ph, params_model)))
        vfadam = tf.train.AdamOptimizer(learning_rate=5e-3, epsilon=1e-7)
        # self._train_vf_grads = vfadam.apply_gradients(list(zip(self.vfg_ph, params_critic)))
        # self._train_pi = adam.minimize(self.policyloss)
        self._train_vf = vfadam.minimize(self.vf_loss)


    def act(self, ob):
        return tf.get_default_session().run([self.sampler], feed_dict={self.st:[ob]})[0][0]

    def act_batch(self, ob_batch):
        return tf.get_default_session().run([self.sampler], feed_dict={self.st:ob_batch})[0]

    def getvf(self, ob_list):
        return tf.get_default_session().run([self.vf], feed_dict={self.st:ob_list})[0]

    def get_pg_loss_h_grads(self, ob_list, ac_list, ret_list):
        return tf.get_default_session().run([self.policyloss, self.entropy, self.policylossgrad],
                feed_dict={self.st:ob_list, self.at:ac_list, self.At:ret_list})

    def train_step(self, grad):
        tf.get_default_session().run([self._train_pi_grads], feed_dict = dict(list(zip(*[self.pg_ph,grad]))))

    def vf_train_step(self, ob_list, vtarg_list):
        return tf.get_default_session().run([self.vf_loss, self._train_vf], feed_dict={self.st:ob_list, self.Vtarg:vtarg_list})

    def train_epoch(self):
        cfg = self.cfg
        env = cfg['env']
        loss_list, grad_list, h_list, eplen_list, eprew_list, vf_loss_list = [], [], [], [], [], []
        for _ in range(self.cfg['neps_iter']):
            ob = env.reset()
            ob_list, ac_list , rew_list, = [], [], []
            rew, done = 0, False
            while not done :
                ob_list.append(ob)
                ac = self.act(ob)
                ac_list.append(ac)
                ob, rew, done, _ = env.step(ac)
                rew_list.append(rew)
            ret_list, ret = [0.0], 0.0
            for rew in rew_list[:-1]:
                ret = rew + cfg['gamma']*ret
                ret_list.append(ret)
            ret_list = ret_list[::-1]
            for _ in range(cfg['vf_iters']):
                inds = np.random.choice(len(ob_list), cfg['vf_batch_size'])
                vf_loss, _ = self.vf_train_step(np.array(ob_list)[inds], np.array(ret_list)[inds])
                vf_loss_list.append(vf_loss)
            vf_list = self.getvf(ob_list)
            adv_list = []
            for i, rew in enumerate(rew_list[:-1]):
                adv = rew + cfg['gamma']*vf_list[i+1] - vf_list[i]
                adv_list.append(adv)
            adv_list.append(0.0)
            pgl, h_ep, grad = self.get_pg_loss_h_grads(ob_list, ac_list, adv_list)
            h_ep = np.mean(h_ep)
            loss_list.append(pgl)
            h_list.append(h_ep)
            grad_list.append(grad)
            eplen_list.append(len(ob_list))
            eprew_list.append(np.sum(rew_list))
        grad_final = [np.mean(g,axis=0) for g in list(map(list, zip(*grad_list)))]
        self.train_step(grad_final)
        return tuple(map(lambda x: np.mean(x), [eplen_list, eprew_list, h_list, loss_list, vf_loss_list]))

    def play(self):
        cfg = self.cfg
        env = cfg['env']
        ob = env.reset()
        trew, done = 0, False
        while not done :
            ac = self.act(ob)
            ob, rew, done, _ = env.step(ac)
            trew += rew
            env.render()
        print(trew)

    def save(self, loc):
        saver = tf.train.Saver()
        save_path = saver.save(sess, loc)
        print("Model saved in path: %s" %save_path)

    def load(self, loc):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), loc)

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    cfg = {'env':env, 'gamma':0.99, 'entcoeff':0.01,
           'vf_iters':4, 'vf_batch_size':8,
           'neps_iter':4}
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.__enter__()
    model = REINFORCE_AC(cfg)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    nepochs, ep_interval = 501, 25
    for epoch in range(nepochs):
        eplen, tot_rew, mh, mpgloss, mvfloss = model.train_epoch()
        if epoch%ep_interval==0:
            print('Epoch %d\nEpisode Length: %d\nTotal Reward: %f\nAverage Entropy: %f\nPolicy Loss: %f\nVF Loss: %f\n'%(epoch, eplen, tot_rew, mh, mpgloss, mvfloss))
    while True :
        uin = input('Number of eps to render or q to quit')
        if(uin.isnumeric()):
            for _ in range(int(uin)):
                model.play()
        elif uin == 'q' :
            break
    saveloc = input('Save checkpoint name (blank to skip):')
    if saveloc :
        model.save('models/'+saveloc+'/model.ckpt')
