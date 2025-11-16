import argparse
import os
from typing import List, Optional

from banditbench.utils import compute_cumulative_reward, plot_cumulative_reward
from mylib.env import MovieLensCBEnv
from banditbench.agents.classics import LinUCBAgent


def run_trial(env: MovieLensCBEnv, agent: LinUCBAgent, seed: Optional[int] = None) -> List[float]:
    """Run a single trial and return rewards per step, using the Gym wrapper."""
    rewards: List[float] = []

    _, info = env.reset(seed=seed)
    agent.reset()

    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Use agent on contextual state via the underlying core bandit
        state = env.current_state  # banditbench State
        action = agent.act(state)
        _, reward, terminated, truncated, step_info = env.step(action)
        agent.update(state, int(action), float(reward), step_info)
        rewards.append(float(reward))
        print("----------------")
        print(step_info["raw_prompt"])
        print(f"reward: {reward:.2f}")
        print(f"LinUCB reward: {step_info['ucb_reward']:.2f}")
        print(f"all_arms_pulled: {step_info['all_arms_pulled']}")
        print(f"largest_count: {step_info['largest_count']}")
        print(f"regret / max_regret: {step_info['regret']:.2f} / {step_info['max_regret']:.2f}")

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate LinUCB on MovieLens contextual bandit (Gym wrapper)")
    parser.add_argument("--task-name", default="100k-ratings", choices=["100k-ratings", "1m-ratings"],
                        help="MovieLens dataset variant to use")
    parser.add_argument("--num-arms", type=int, default=5, help="Number of top movies as arms")
    parser.add_argument("--horizon", type=int, default=300, help="Number of interactions per trial")
    parser.add_argument("--rank-k", type=int, default=5, help="SVD rank for features (<= num_arms)")
    parser.add_argument("--mode", default="train", choices=["train", "eval"], help="Context sampling split")
    parser.add_argument("--alpha", type=float, default=0.5, help="LinUCB exploration parameter")
    parser.add_argument("--trials", type=int, default=64, help="Number of independent trials")
    parser.add_argument("--plot-file", default="linucb_movielens.png",
                        help="Optional path to save a cumulative reward plot (png/pdf)")

    args = parser.parse_args()

    # Build environment wrapper
    env = MovieLensCBEnv(
        task_name=args.task_name,
        num_arms=args.num_arms,
        horizon=args.horizon,
        rank_k=args.rank_k,
        mode=args.mode,
        seed=0,
        linucb_alpha=args.alpha,
    )

    # Build banditbench LinUCB agent using the underlying contextual bandit
    agent = LinUCBAgent(env.core_bandit, alpha=args.alpha)

    all_rewards: List[List[float]] = []

    print(
        f"Running LinUCB (Gym wrapper) on MovieLens: task={args.task_name}, arms={args.num_arms}, horizon={args.horizon}, "
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
