import gym
import numpy as np
import random
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv

def set_global_seeds(seed):
  """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
#   if tf is not None:
#     if hasattr(tf.random, 'set_seed'):
#       tf.random.set_seed(seed)
#     elif hasattr(tf.compat, 'v1'):
#       tf.compat.v1.set_random_seed(seed)
#     else:
#       tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # prng was removed in latest gym version
  if hasattr(gym.spaces, 'prng'):
    gym.spaces.prng.seed(seed)

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    #params['reward_type'] = env._kwargs.reward_type

    print('Env observation dimension: {}'.format(params['obs']))
    print('Env goal dimension: {}'.format(params['goal']))
    print('Env action dimension: {}'.format(params['action']))
    print('Env max action value: {}'.format(params['action_max']))
    print('Env max timestep value: {}'.format(params['max_timesteps']))
    return params

# TODO Do we need any wrapper or they are directly usable.
def make_vec_env(env_id, num_envs, seed):
    # ensure each environment is initialised with a different seed.
    env_fns = [make_env_by_id(env_id, seed, rank) for rank in range(num_envs)]

    # TODO (lisheng) Check with MHER/baselines codes.
    if num_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)

    # TODO(lisheng) Obtain the environment information.    
    return env

def make_env_by_id(env_id, seed, rank):
    def _init():
        # currently, we only support the six envs used in the benchmarks
        env = gym.make(env_id)
        env.seed(seed + rank)
        set_global_seeds(seed + rank)
        return env
    return _init
   

# we require an additional compute reward function.
# a simplified version from the mrl repository
class GCVecEnv:
    def __init__(self, env_id, num_envs, seed):
        self.num_envs = num_envs
        sample_env = make_env_by_id(env_id, seed, 0)()
        self.env = make_vec_env(env_id, num_envs, seed)
        self.env_params = get_env_params(sample_env)
        try:
            self.compute_reward = sample_env.compute_reward
        except:
            raise ValueError("env %s doesn't have compute reward \
                function" % env_id)
        self.state = self.env.reset()
    
    def step(self, action):
        res = self.env.step(action)
        self.state = res[0]
        return res
    
    def reset(self, indices=None):
        if not indices:
            self.state = self.env.reset()
            return self.state
        else:
            reset_states = self.env.env_method('reset', indices=indices)
            for i, reset_state in zip(indices, reset_states):
                for key in reset_state:
                    self.state[key][i] = reset_state[key]
            return self.state