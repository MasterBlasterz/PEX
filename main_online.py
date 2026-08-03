import itertools
import os
import torch
from tqdm import trange

import random
import wandb

from pex.algorithms.pex_grouped import PEXGrouped
from pex.algorithms.pex import PEX
from pex.algorithms.iql_online import IQL_online
from pex.networks.policy import GaussianPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork
from pex.utils.util import (
    set_seed, ReplayMemory, torchify, eval_policy, torchify, DEFAULT_DEVICE,
    get_batch_from_dataset_and_buffer,
    eval_policy, set_default_device, get_env_and_dataset)


HUMANOID_JOINT_GROUPS = {
    "abdomen": [0, 1, 2],                           # torso
    "hip": [3, 4, 5, 7, 8, 9],                      # hips 
    "knee": [6, 10],                                # knees
    "shoulder": [11, 12, 14, 15],                   # shoulders
    "elbow": [13, 16],                              # elbows
}


def main(args):
    torch.set_num_threads(1)

    if os.path.exists(args.log_dir):
        print(f"The directory {args.log_dir} exists. Please specify a different one.")
        return
    else:
        print(f"Creating directory {args.log_dir}")
        os.mkdir(args.log_dir)


    env, dataset, reward_transformer = get_env_and_dataset(args.env_name, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]


    if args.seed is None:
        args.seed = random.randrange(2**32)
    set_seed(args.seed, env=env)

    if torch.cuda.is_available():
        set_default_device()

    action_space = env.action_space
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)

    algorithm_option = args.algorithm.upper()

    if algorithm_option == "SCRATCH":
        double_buffer = False
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )

    elif algorithm_option == "BUFFER":
        double_buffer = True
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )

    elif algorithm_option == "DIRECT":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path
        )

    elif algorithm_option == "PEX":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = PEX(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
        )
    elif algorithm_option == "PEX-GROUPED":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = PEXGrouped(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
        )
    elif algorithm_option == "POLICY-TRANSFER":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            copy_policy=False
        )

    wandb.init(
        entity="fryan-nr",
        project="NR3",
        name=f"NR3_{args.env_name}_{args.seed}_online_{args.algorithm}",
        tags=["NR3", "calql", args.env_name, "medium-replay", str(args.seed)],
        config={
            "env_name": args.env_name,
            "seed": args.seed,
            "algorithm": args.algorithm,
            "inv_temperature": args.inv_temperature,
            "discount": args.discount,
            "hidden_dim": args.hidden_dim,
            "hidden_num": args.hidden_num,
            "total_env_steps": args.total_env_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "target_update_rate": args.target_update_rate,
            "tau": args.tau,
            "beta": args.beta,
            "eval": args.eval,
            "eval_period": args.eval_period,
            "eval_episode_num": args.eval_episode_num,
            "max_episode_steps": args.max_episode_steps,
            "replay_size": args.replay_size,
            "initial_collection_steps": args.initial_collection_steps,
            "updates_per_step": args.updates_per_step,
            "replay_size": args.replay_size,
            "ckpt_path": args.ckpt_path,
            "is_cuda": torch.cuda.is_available(),
            "log_dir": args.log_dir,
            "ablation": args.ablation,
        },
    )

    memory = ReplayMemory(args.replay_size, args.seed)

    total_numsteps = 0
    fall_count = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, info = env.reset()

        # jk59: per episode counter
        offline_policy_count = 0
        online_policy_count = 0

        if "PEX-GROUPED" in algorithm_option:
            offline_policy_count_grouped = dict.fromkeys(HUMANOID_JOINT_GROUPS.keys(), 0)
            online_policy_count_grouped = dict.fromkeys(HUMANOID_JOINT_GROUPS.keys(), 0)

        while not done:
            if "PEX" in algorithm_option:
                action, choice = alg.select_action(
                    torchify(state).to(DEFAULT_DEVICE),
                    return_policy_selection=True, ablation=args.ablation
                )
                
                if "PEX-GROUPED" in algorithm_option:
                    for i, (group_name, indices) in enumerate(HUMANOID_JOINT_GROUPS.items()):
                        offline_policy_count_grouped[group_name] += choice[i] == 0
                        online_policy_count_grouped[group_name] += choice[i] != 0
                        offline_policy_count += sum(count == 0 for count in choice) / len(choice)
                        online_policy_count += sum(count != 0 for count in choice) / len(choice)
                else:
                    offline_policy_count += choice == 0
                    online_policy_count += choice != 0

            else:
                action = alg.select_action(torchify(state).to(DEFAULT_DEVICE))
            action = action.detach().cpu().numpy()

            if len(memory) > args.initial_collection_steps:
                for i in range(args.updates_per_step):
                    alg.update(*get_batch_from_dataset_and_buffer(dataset, memory, args.batch_size, double_buffer))

            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated:
                fall_count += 1
            done = terminated or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            reward_for_replay = reward_transformer(reward)


            terminal = 0 if episode_steps == env._max_episode_steps else float(done)
            memory.push(state, action, reward_for_replay, next_state, terminal)
            state = next_state

            if total_numsteps % args.eval_period == 0 and args.eval is True:
                print("Episode: {}, total env-steps: {}".format(i_episode, total_numsteps))
                eval_metrics = eval_policy(env, args.env_name, alg, args.max_episode_steps, args.eval_episode_num)
                eval_metrics['fall_count_training'] = fall_count / episode_steps * 100.0

                if "PEX-GROUPED" in algorithm_option:
                    for group_name in HUMANOID_JOINT_GROUPS.keys():
                        eval_metrics[f'pex/offline_policy_percentage_{group_name}'] = (offline_policy_count_grouped[group_name] / episode_steps) * 100.0
                        eval_metrics[f'pex/online_policy_percentage_{group_name}'] = (online_policy_count_grouped[group_name] / episode_steps) * 100.0
                if "PEX" in algorithm_option:
                    offline_pct = (offline_policy_count / episode_steps) * 100.0
                    online_pct = (online_policy_count / episode_steps) * 100.0

                    eval_metrics['pex/offline_policy_select_count'] = offline_policy_count
                    eval_metrics['pex/online_policy_select_count'] = online_policy_count
                    eval_metrics['pex/offline_policy_percentage'] = offline_pct
                    eval_metrics['pex/online_policy_percentage'] = online_pct
                    eval_metrics['train/episode_reward'] = episode_reward
                    eval_metrics['train/episode_steps'] = episode_steps

                if eval_metrics is not None:
                    wandb.log(
                        {f"eval/{k}": v for k, v in eval_metrics.items()},
                        step=total_numsteps,
                    )

        if total_numsteps > args.total_env_steps:
            break

        env.close()

    torch.save(alg.state_dict(), args.log_dir + '/{}_online_ckpt'.format(args.algorithm))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--algorithm', required=True)  # ['direct', 'buffer', 'pex', 'pex-grouped', 'policy-transfer', 'scratch']
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0,
                        help='IQL inverse temperature')
    parser.add_argument('--ckpt_path', default=None,
                    help='path to the offline checkpoint')

    parser.add_argument('--replay_size', type=int, default=1_000_000, metavar='N',
                        help='size of replay buffer (default: 1_000_000)')
    parser.add_argument('--total_env_steps', type=int, default=1_000_001, metavar='N',
                        help='total number of env steps (default: 1_000_000)')
    parser.add_argument('--initial_collection_steps', type=int, default=5_000, metavar='N',
                        help='Initial environmental steps before training starts (default: 5_000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--inv_temperature', type=float, default=3, metavar='G',
                        help='inverse temperature for PEX action selection (default: 3)')
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--eval_period', type=int, default=10_000)
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1_000)
    parser.add_argument('--ablation', type=str, default='none', choices=['none', 'uni', 'unfreeze-policy'],)

    main(parser.parse_args())
