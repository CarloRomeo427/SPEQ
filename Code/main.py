import os

# os.environ["MUJOCO_PY_MUJOCO_PATH"] = "/andromeda/personal/gmacaluso/mujoco210/bin"
# os.environ["LD_LIBRARY_PATH"] = "/andromeda/personal/gmacaluso/mujoco210/bin/bin"
# os.environ["WANDB_MODE"] = 'offline'
import os.path

import gym
import numpy as np
import torch
import time
import sys

import wandb

from SPEQ.Code.speq.algos.agent import Agent
from speq.algos.core import mbpo_epoches, test_agent
from speq.utils.run_utils import setup_logger_kwargs
from speq.utils.bias_utils import log_bias_evaluation
from speq.utils.logx import EpochLogger

import customenvs

customenvs.register_mbpo_environments()


def print_class_attributes(obj):
    """
    Prints all attributes of an object along with their values.

    Parameters:
    obj (object): The object whose attributes need to be printed.
    """
    attributes = vars(obj)
    for attr, value in attributes.items():
        print(f"{attr}: {value}")


def speq(env_name, seed=0, epochs='mbpo', steps_per_epoch=1000,
             max_ep_len=1000, n_evals_per_epoch=1,
             logger_kwargs=dict(), debug=False,
             hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
             lr=3e-4, gamma=0.99, polyak=0.995,
             alpha=0.2, auto_alpha=True, target_entropy='mbpo',
             start_steps=5000, delay_update_steps='auto',
             utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
             policy_update_delay=20,
             evaluate_bias=False, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
             gpu_id=0, target_drop_rate=0.0, layer_norm=False,
             method="sac", offline_frequency=1000,
             offline_epochs=100, ):

    if debug: 
        hidden_sizes = [2, 2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

    assert method in ["sac", "redq", "duvn", "monosac"], "illigal method:" + str(method)
    if method == "sac":
        print("[MAIN]: use SAC. set num_Q to 2")
        num_Q = 2
    elif method == "duvn":
        print("[MAIN]: use DUVN. set num_Q  and  num_min to 1")
        num_Q = 1
        num_min = 1

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    if epochs == 'mbpo' or epochs < 0:
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        mbpo_epoches['HumanoidTruncatedObs-v2'] = 300
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: gym.make(args.env)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()

    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)

    seed_all(epoch=0)

    """prepare to init agent"""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    act_limit = env.action_space.high[0].item()
    start_time = time.time()
    sys.stdout.flush()

    """init agent and start training"""
    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                         hidden_sizes, replay_size, batch_size,
                         lr, gamma, polyak,
                         alpha, auto_alpha, target_entropy,
                         start_steps, delay_update_steps,
                         utd_ratio, num_Q, num_min, q_target_mode,
                         policy_update_delay,
                         target_drop_rate=target_drop_rate,
                         layer_norm=layer_norm, 
                         policy_polyak_update=policy_polyak_update,
                         )

    print_class_attributes(agent)

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):

        a = agent.get_exploration_action(o, env)

        o2, r, d, _ = env.step(a)

        ep_len += 1
        d = False if ep_len == max_ep_len else d  

        agent.store_data(o, a, r, o2, d)

        if offline_frequency > 0 and (t + 1) % offline_frequency == 0:
            agent.finetune_offline(epochs=offline_epochs, test_env=test_env)


        agent.train(logger)

        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):

            logger.store(EpRet=ep_ret, EpLen=ep_len)

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        

        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            test_rw = test_agent(agent, test_env, max_ep_len, logger)  # add logging here
            wandb.log({"EvalReward": np.mean(test_rw)})

            if evaluate_bias:
                normalized_bias_sqr_per_state, normalized_bias_per_state, bias = log_bias_evaluation(bias_eval_env,
                                                                                                     agent, logger,
                                                                                                     max_ep_len, alpha,
                                                                                                     gamma, n_mc_eval,
                                                                                                     n_mc_cutoff)
                wandb.log({"normalized_bias_sqr_per_state": np.abs(np.mean(normalized_bias_sqr_per_state)),
                           "normalized_bias_per_state": np.mean(normalized_bias_per_state),
                           "bias": np.mean(bias)
                           })


            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()


            sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='Hopper-v2')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=-1) 
    parser.add_argument('-exp_name', type=str, default='speq')
    parser.add_argument('-debug', action='store_true')


    parser.add_argument("-info", type=str, help="folder of the experiments")
    parser.add_argument("-gpu_id", type=int, default=0,
                        help="GPU device ID to be used in GPU experiment, default is 1e6")
    parser.add_argument("-method", default="sac", choices=["sac", "redq", "duvn", "monosac"],
                        help="method, default=sac")
    parser.add_argument("-target_drop_rate", type=float, default=0.0,
                        help="drop out rate of target value_net function, default=0")
    parser.add_argument("-layer_norm", type=int, default=1, choices=[0, 1],
                        help="Using layer normalization for training critics if set to 1 (TH), default=0")
    parser.add_argument("-offline_frequency", type=int, default=75000, )
    parser.add_argument("-offline_epochs", type=int, default=10000, )
    parser.add_argument("-utd", type=int, default=1, )
    parser.add_argument("-network_width", type=int, default=256, )
    parser.add_argument("-num-q", type=int, default=2, )
    parser.add_argument("-evaluate_bias", default=False, action='store_true')

    args = parser.parse_args()


    exp_name_full = args.exp_name + '_%s' % args.env


    args.data_dir = './runs/' + str(args.info) + '/'



    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    hidden_sizes = (args.network_width, args.network_width)

    wandb.init(

        name=f'{exp_name_full}',
        project="SPEQ",
        group=args.env,

        config={
            "seed": args.seed,
            "offline_frequency": args.offline_frequency,
            "offline_epochs": args.offline_epochs,
            "utd_ratio_online": args.utd_ratio_online,
            "network_width": args.network_width,
            "evaluate_bias": args.evaluate_bias,
        })

    speq(args.env, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs, debug=args.debug,
             gpu_id=args.gpu_id,
             target_drop_rate=args.target_drop_rate, 
             layer_norm=bool(args.layer_norm),
             offline_frequency=args.offline_frequency, num_Q=args.num_q,
             offline_epochs=args.offline_epochs, 
             method=args.method,
             utd_ratio=args.utd_ratio_online, hidden_sizes=hidden_sizes,
             evaluate_bias=args.evaluate_bias)
