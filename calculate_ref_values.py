"""
Estimate ref_min_score (random policy) and ref_max_score (expert policy)
for the Gymnasium Humanoid-v5 environment, matching the convention used
by Minari / D4RL for offline RL score normalization:

    normalized_score = 100 * (score - ref_min_score) / (ref_max_score - ref_min_score)

Usage:
    pip install gymnasium[mujoco] minari --break-system-packages

    # Random-policy baseline only:
    python compute_humanoid_ref_scores.py --episodes 50

    # Recommended: pull the expert reference straight from a Minari dataset's
    # recorded returns (no extra ML deps, no model file needed):
    python compute_humanoid_ref_scores.py --episodes 50 \\
        --expert-minari-dataset mujoco/humanoid/expert-v0

    # Alternative: rollout with an actual trained model file
    # (requires sb3-contrib, since TQC is not in core stable-baselines3):
    pip install sb3-contrib --break-system-packages
    python compute_humanoid_ref_scores.py --episodes 50 --expert-model path/to/model.zip
"""

import argparse
import numpy as np
import gymnasium as gym


def run_episodes(env, policy_fn, num_episodes, seed=0):
    """Run num_episodes and return the list of undiscounted episode returns."""
    returns = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    return returns


def random_policy_fn(env):
    def _fn(obs):
        return env.action_space.sample()
    return _fn


def load_expert_policy_fn(model_path):
    """Load a Stable-Baselines3 model and wrap it as a policy function.

    Note: TQC is not part of core stable-baselines3, it's in the
    sb3-contrib package. Install with:
        pip install sb3-contrib --break-system-packages
    (The PyPI package literally named "tqc" is unrelated and will NOT work here.)
    """
    from sb3_contrib import TQC
    model = TQC.load(model_path)

    def _fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return _fn


def ref_max_from_minari_dataset(dataset_id):
    """Compute ref_max_score directly from a Minari dataset's recorded episode
    returns, instead of re-running rollouts with a loaded model. This is the
    simplest option if you just want the reference score for an existing
    Minari expert/medium/simple dataset (e.g. 'mujoco/humanoid/expert-v0').
    """
    import minari
    dataset = minari.load_dataset(dataset_id, download=True)
    episode_returns = [float(np.sum(ep.rewards)) for ep in dataset.iterate_episodes()]
    return float(np.mean(episode_returns)), episode_returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Humanoid-v5",
                         help="Gymnasium environment id (default: Humanoid-v5)")
    parser.add_argument("--episodes", type=int, default=50,
                         help="Number of episodes to average over per policy")
    parser.add_argument("--expert-model", default=None,
                         help="Path to a trained SB3/sb3-contrib model .zip file (optional). "
                              "Requires sb3-contrib for TQC models.")
    parser.add_argument("--expert-minari-dataset", default=None,
                         help="Minari dataset id to pull expert returns from directly, "
                              "e.g. 'mujoco/humanoid/expert-v0'. Simplest option — no "
                              "policy rollout or extra ML dependencies needed.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(args.env_id)

    # --- Random policy: ref_min_score ---
    print(f"Running random policy on {args.env_id} for {args.episodes} episodes...")
    random_returns = run_episodes(env, random_policy_fn(env), args.episodes, seed=args.seed)
    ref_min_score = float(np.mean(random_returns))
    print(f"  ref_min_score (random policy mean return): {ref_min_score:.2f}")
    print(f"  std: {np.std(random_returns):.2f}, "
          f"min: {np.min(random_returns):.2f}, max: {np.max(random_returns):.2f}")

    ref_max_score = None
    if args.expert_minari_dataset is not None:
        # --- Expert policy: ref_max_score, read directly from a Minari dataset ---
        print(f"\nReading expert returns from Minari dataset "
              f"'{args.expert_minari_dataset}'...")
        ref_max_score, expert_returns = ref_max_from_minari_dataset(args.expert_minari_dataset)
        print(f"  ref_max_score (expert dataset mean episode return): {ref_max_score:.2f}")
        print(f"  n_episodes: {len(expert_returns)}, "
              f"std: {np.std(expert_returns):.2f}, "
              f"min: {np.min(expert_returns):.2f}, max: {np.max(expert_returns):.2f}")
    elif args.expert_model is not None:
        # --- Expert policy: ref_max_score, via rollout with a loaded SB3/sb3-contrib model ---
        print(f"\nRunning expert policy ({args.expert_model}) "
              f"on {args.env_id} for {args.episodes} episodes...")
        expert_fn = load_expert_policy_fn(args.expert_model)
        expert_returns = run_episodes(env, expert_fn, args.episodes, seed=args.seed + 1000)
        ref_max_score = float(np.mean(expert_returns))
        print(f"  ref_max_score (expert policy mean return): {ref_max_score:.2f}")
        print(f"  std: {np.std(expert_returns):.2f}, "
              f"min: {np.min(expert_returns):.2f}, max: {np.max(expert_returns):.2f}")

    env.close()

    print("\n--- Summary ---")
    print(f"ref_min_score = {ref_min_score:.2f}")
    if ref_max_score is not None:
        print(f"ref_max_score = {ref_max_score:.2f}")
        print(f"ref_max - ref_min = {ref_max_score - ref_min_score:.2f}")
    else:
        print("ref_max_score not computed (pass --expert-model to include it).")
        print("\nAlternatively, if you already loaded a Minari dataset, you can reuse")
        print("its recorded episode returns directly as your expert reference instead")
        print("of re-running rollouts:")
        print("""
    import minari
    import numpy as np

    dataset = minari.load_dataset("mujoco/humanoid/expert-v0")
    episode_returns = [ep.rewards.sum() for ep in dataset.iterate_episodes()]
    ref_max_score = float(np.mean(episode_returns))
""")


if __name__ == "__main__":
    main()