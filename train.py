import random
import numpy as np

import gym
import torch

from env import GCVecEnv
from arguments import get_args
from ddpg_agent import ddpg_agent

def launch(args):
    # create the ddpg_agent
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # env = gym.make(args.env_name)
    # NOTE It's just a simple vectorised/parallelised env without any wrapper.
    env = GCVecEnv(args.env_name, args.num_envs, args.seed)
    # set random seeds for reproduce
    # get the environment parameters
    # env_params = get_env_params(env,args)

    print('Run training with seed {}'.format(args.seed))
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env.env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)
