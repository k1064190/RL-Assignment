import os
import time
import argparse
import pickle
from collections import OrderedDict

import gym
import numpy as np
import torch

from aai4160.infrastructure.bc_trainer import BCTrainer
from aai4160.agents.bc_agent import BCAgent
from aai4160.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from aai4160.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES

from aai4160.infrastructure import pytorch_util as ptu
from aai4160.infrastructure import utils
from aai4160.infrastructure.logger import Logger

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_policy_file", type=str, default="aai4160/policies/experts/Ant.pkl")
    parser.add_argument('--env_name', '-env', type=str,
        help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--ep_len', type=int)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # load expert policy
    args = parseArgs()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(
        use_gpu=not args.no_gpu,
        gpu_id=args.which_gpu
    )

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../../data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = 'ex_' + args.env_name + '_' + \
        time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logger = Logger(logdir)
    env = gym.make(args.env_name, **MJ_ENV_KWARGS[args.env_name])
    env.reset(seed=args.seed)

    print('Loading expert policy from...', args.expert_policy_file)
    loaded_expert_policy = LoadedGaussianPolicy(
        args.expert_policy_file).to('cpu')
    print('Done restoring expert policy...')

    # collect eval trajectories, for logging
    print("\nCollecting data for eval...")
    eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
        env, loaded_expert_policy, args.eval_batch_size, args.ep_len)

    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    print("\nEval data collected:")
    logs = OrderedDict()

    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    for key, value in logs.items():
        print('{} : {}'.format(key, value))
        logger.log_scalar(value, key, 0)

    logger.flush()

    print("Done!")




