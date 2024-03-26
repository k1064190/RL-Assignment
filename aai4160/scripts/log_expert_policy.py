import os
import time
import argparse

from aai4160.infrastructure.bc_trainer import BCTrainer
from aai4160.agents.bc_agent import BCAgent
from aai4160.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from aai4160.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES

from aai4160.infrastructure import pytorch_util as ptu

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_policy_file", type=str, default="aai4160/policies/experts/Ant.pkl")
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
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

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../../data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = 'ex' + args.exp_name + '_' + args.env_name + '_' + \
        time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

