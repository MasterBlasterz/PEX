import argparse
import torch

from pex.algorithms.iql_online import IQL_online
from pex.networks.policy import GaussianPolicy

from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork
from pex.utils.util import ( eval_policy,
    eval_policy, get_env_and_dataset)


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", required=True,
                         help="Minari dataset id, e.g. 'D4RL/door/expert-v2'")
    parser.add_argument("--model_path", required=True,
                         help="Path to the trained model (.zip) to evaluate at "
                              "timestep 1,000,000")
    parser.add_argument("--eval_period", type=int, default=10,
                         help="Number of episodes to average over per timestep")
    parser.add_argument("--max_episode_steps", type=int, default=1_000,
                         help="Max steps per episode (default: 1,000)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                         help="Use deterministic actions during evaluation")
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0,
                        help='IQL inverse temperature')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--ckpt_path', default=None,
                    help='path to the offline checkpoint')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--eval_episode_num', type=int, default=50,
                        help='Number of evaluation episodes (default: 10)')
    args = parser.parse_args()

    import wandb

    print(f"[minari] loading dataset '{args.env_name}' and recovering environment...")
    env, dataset, reward_transformer = get_env_and_dataset(args.env_name, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    action_space = env.action_space
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)

    alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.model_path
        )

    env.reset(seed=args.seed)

    wandb.init(
        entity="fryan-nr",
        project="NR3",
        name=f"NR3_{args.env_name}_{args.seed}_offline_baseline",
        tags=["NR3", "calql", args.env_name, "medium-replay", str(args.seed)],
        config={
            "env_name": args.env_name,
            "seed": args.seed,
            "eval_steps": args.eval_period,
            "ckpt_path": args.model_path,
            "algorithm": "Offline",
        },
    )

    result = eval_policy(env, args.env_name, alg, args.max_episode_steps, args.eval_episode_num)

    for i in [1, 1_000, 10_000, 100_000, 1_000_000]:
        wandb.log(
            {
                "eval/return mean": result["return mean"],
                "eval/return std": result["return std"],
                'eval/normalized return mean': result["normalized return mean"],
                'eval/normalized return std': result["normalized return std"],
                "eval/n_episodes": args.eval_episode_num,
            },
            step=i,
        )

    env.close()
    print("[done] logged evaluation results to wandb.")
    


if __name__ == "__main__":
    main()
