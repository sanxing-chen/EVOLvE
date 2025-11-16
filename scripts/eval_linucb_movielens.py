import argparse
import os
from typing import List, Optional

import numpy as np

from banditbench.tasks.cb.movielens import MovieLens
from banditbench.agents.classics import LinUCBAgent
from banditbench.utils import compute_cumulative_reward, plot_cumulative_reward


def run_trial(env: MovieLens, agent: LinUCBAgent, seed: Optional[int] = None) -> List[float]:
    """Run a single trial and return rewards per step."""
    rewards: List[float] = []

    state, _ = env.reset(seed=seed)
    agent.reset()

    done = False
    while not done:
        action = agent.act(state)
        new_state, reward, done, info = env.step(state, action)
        agent.update(state, action, reward, info)
        rewards.append(float(reward))
        state = new_state

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate LinUCB on MovieLens contextual bandit")
    parser.add_argument("--task-name", default="100k-ratings", choices=["100k-ratings", "1m-ratings"],
                        help="MovieLens dataset variant to use")
    parser.add_argument("--num-arms", type=int, default=5, help="Number of top movies as arms")
    parser.add_argument("--horizon", type=int, default=300, help="Number of interactions per trial")
    parser.add_argument("--rank-k", type=int, default=5, help="SVD rank for features (<= num_arms)")
    parser.add_argument("--mode", default="train", choices=["train", "eval"], help="Context sampling split")
    parser.add_argument("--alpha", type=float, default=0.5, help="LinUCB exploration parameter")
    parser.add_argument("--trials", type=int, default=64, help="Number of independent trials")
    parser.add_argument("--save-data-dir", default="./tensorflow_datasets/",
                        help="Directory for TFDS cache (must contain MovieLens)")
    parser.add_argument("--plot-file", default="linucb_movielens.png",
                        help="Optional path to save a cumulative reward plot (png/pdf)")

    args = parser.parse_args()

    # Hint to user about TFDS cache location
    if args.save_data_dir and not os.path.isdir(args.save_data_dir):
        print(f"Warning: save_data_dir '{args.save_data_dir}' does not exist. TFDS will try default cache.")

    # Build environment
    env = MovieLens(
        task_name=args.task_name,
        num_arms=args.num_arms,
        horizon=args.horizon,
        rank_k=args.rank_k,
        mode=args.mode,
        seed=0,
        save_data_dir=args.save_data_dir,
    )

    # Build agent
    agent = LinUCBAgent(env, alpha=args.alpha)

    all_rewards: List[List[float]] = []

    print(
        f"Running LinUCB on MovieLens: task={args.task_name}, arms={args.num_arms}, horizon={args.horizon}, "
        f"rank_k={args.rank_k}, mode={args.mode}, alpha={args.alpha}"
    )

    for i in range(args.trials):
        print(f"Trial {i}/{args.trials}...")
        rewards = run_trial(env, agent, seed=i)
        all_rewards.append(rewards)

    # Aggregate and report
    reward_means, reward_sems = compute_cumulative_reward(all_rewards, args.horizon)
    final_mean = reward_means[-1]
    final_sem = reward_sems[-1]

    print("\nResults:")
    print(f"- Final average reward over horizon: {final_mean:.4f} ± {final_sem:.4f}")

    plot_cumulative_reward(all_rewards, args.horizon,
                           title=f"LinUCB on MovieLens ({args.task_name}, k={args.num_arms})",
                           filename=args.plot_file)


if __name__ == "__main__":
    main()

