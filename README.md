This is a standalone implementation of [DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) in Tensorflow. Running `dagger.py` trains a NN policy through behavioral cloning of an expert agent on the [CartPole](https://gym.openai.com/envs/CartPole-v1/) OpenAI Gym environment. The expert agent is automatically loaded from the `models` folder.

Best results were obtained with large data aggregation buffer sizes (~25000), and were in fact necessary for convergence.

### Requirements

* OpenAI Gym
* Tensorflow (v1.3 or newer)

### Using custom expert agents

The expert agent is supplied through the `'expert_model'`  key in the `cfg` dictionary used to create the `Dagger` object. It needs to expose only one method, `act_batch(ob_list)` which takes in a list of observations of compatible shape and returns the corresponding expert labels.
