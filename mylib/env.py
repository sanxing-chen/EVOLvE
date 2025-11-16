from math import sqrt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple
from typing import Dict, Any, Optional

from banditbench.tasks.cb.movielens.env import MovieLens, MovieLensVerbal
from banditbench.agents.classics import LinUCBAgent

class Bandit(gym.Env):
    """
    Base class for bandit environments that handles common functionality like
    tracking history and generating prompts.
    """
    def __init__(self, n_arms: int, max_steps: int, template_type: str = 'base'):
        super().__init__()
        self.n_arms = n_arms
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Discrete(1)  # Dummy observation space
        self.history = []
        self.template_type = template_type
        self.last_info = {}  # Add cache for last info
        self.steps = 0  # Track current step count
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.history = []
        self.steps = 0  # Reset step counter

        # draw max_steps from uniform distribution between 10 and 100
        if self.max_steps < 0:
            self.max_steps = self.np_random.integers(10, 150)
        
        if self.n_arms < 0:
            self.n_arms = self.np_random.integers(3, 10)

        self.ucb = UCB(self.n_arms)  # Add UCB tracker
        self.ucb.reset(seed=seed)  # Reset UCB tracker
        
        # Initialize info with initial state
        self.last_info = {
            'regret': 0,  # No regret at start
            'raw_prompt': make_raw_prompt(self.n_arms, self.history, self.template_type),
            'ucb_action': self.ucb.select_action(),
            'ucb_reward': 0,
            'all_arms_pulled': False,
            'largest_count': 0,
            'max_regret': 1,
            }
        return 0, self.last_info
        
    def step(self, action):
        if action >= self.n_arms or not self.action_space.contains(action):
            return 0, 0, False, False, self.last_info
            
        reward = self._get_reward(action)
        self.history.append((action, reward))
        self.steps += 1  # Increment step counter
        
        self.ucb.update(action, reward)
        ucb_action = self.ucb.select_action()

        # compute UCB-based reward separately
        ucb_reward = float(action == self.last_info['ucb_action'])
        
        # Check if we've reached max steps
        done = self.steps >= self.max_steps
        
        regret = self.optimal_value - self._get_mean_reward(action)
        
        # Check if all arms have been pulled at least once
        pulled_arms = set(action for action, _ in self.history)
        all_arms_pulled = len(pulled_arms) == self.n_arms
        
        self.last_info = {
            'regret': regret,
            'raw_prompt': make_raw_prompt(self.n_arms, self.history, self.template_type),
            'ucb_action': ucb_action,
            'ucb_reward': ucb_reward,
            'all_arms_pulled': all_arms_pulled,
            'largest_count': self.ucb.counts.max(),
            'max_regret': self.max_regret,
        }
        return 0, reward, done, False, self.last_info
    
    def _get_reward(self, action):
        """Get reward for the given action. To be implemented by subclasses."""
        raise NotImplementedError
        
    def _get_mean_reward(self, action):
        """Get mean reward for the given action. To be implemented by subclasses."""
        raise NotImplementedError

class BernoulliBandit(Bandit):
    """
    A simple N-armed Bernoulli bandit environment.
    Each arm returns a reward of 1 with probability p, and 0 with probability 1-p.
    The probabilities are resampled on each reset.
    """
    def __init__(self, n_arms=10, max_steps=100, probabilities=None, template_type: str = 'base'):
        super().__init__(n_arms, max_steps, template_type)
        self.fixed_probabilities = None if probabilities is None else np.array(probabilities)
        self.reset()
            
    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        
        # If fixed probabilities provided, use those, otherwise randomly generate
        if self.fixed_probabilities is None:
            self.probabilities = self.np_random.uniform(0, 1, self.n_arms)
        else:
            self.probabilities = self.fixed_probabilities
        
        self.optimal_arm = np.argmax(self.probabilities)
        self.optimal_value = self.probabilities[self.optimal_arm]
        self.max_regret = self.optimal_value - np.min(self.probabilities)
        return obs, info
        
    def _get_reward(self, action):
        return float(self.np_random.random() < self.probabilities[action])
        
    def _get_mean_reward(self, action):
        return self.probabilities[action]

class GaussianBandit(Bandit):
    """
    N-armed Gaussian bandit environment.
    Each arm returns a reward drawn from N(mean, sigma^2) distribution.
    The means are drawn from N(0, sigma^2) and all arms have std=sigma.
    The means are resampled on each reset.
    """
    def __init__(self, n_arms=10, max_steps=100, sigma=1.0, template_type: str = 'base'):
        super().__init__(n_arms, max_steps, template_type)
        self.sigma = sigma
        self.reset()
        
    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        
        # Draw means from N(0, sigma^2)
        self.means = self.np_random.normal(0, sqrt(self.sigma), self.n_arms)

        # Draw means from uniform distribution between 0 and 1
        # self.means = self.np_random.uniform(0, 1, self.n_arms)
            
        self.optimal_arm = np.argmax(self.means)
        self.optimal_value = self.means[self.optimal_arm]
        self.max_regret = self.optimal_value - np.min(self.means)
        return obs, info
        
    def _get_reward(self, action):
        return float(self.np_random.normal(self.means[action], sqrt(self.sigma)))
        
    def _get_mean_reward(self, action):
        return self.means[action]

class UCB():
    """Upper Confidence Bound bandit algorithm"""
    def __init__(self, n_arms: int, c: float = 0.5):
        self.c = c
        self.n_arms = n_arms
        
    def reset(self, seed=None):
        self.q_values = np.zeros(self.n_arms)
        self.counts = np.zeros(self.n_arms)
        self.t = 0
        
    def select_action(self) -> int:
        self.t += 1
        if np.any(self.counts == 0):
            return np.where(self.counts == 0)[0][0]
            
        ucb = self.q_values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb)
    
    def update(self, action: int, reward: float):
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]


    
def format_history(n_arms, history: List[Tuple[int, float]], summarized_history: bool = False) -> str:
    """Format a history of actions and rewards with customizable prefix and title.
    
    Args:
        n_arms: Number of arms
        history: List of (action, reward) tuples
        summarized_history: Whether to summarize the history
        
    Returns:
        Formatted history string
    """
    history_str = ""
    
    if summarized_history:
        # Create a summary of pulls and rewards for each arm
        arm_pulls = {i: [] for i in range(n_arms)}
        for action, reward in history:
            arm_pulls[action].append(reward)
        
        # Format the summary
        for arm in range(n_arms):
            pulls = arm_pulls[arm]
            n_pulls = len(pulls)
            if n_pulls > 0:
                avg_reward = sum(pulls) / n_pulls
                history_str += f"Arm {arm}: {n_pulls} pulls, average reward {avg_reward:.3f}\n"
            else:
                history_str += f"Arm {arm}: never pulled\n"
    else:
        # Original sequential format
        for i, (action, reward) in enumerate(history):
            history_str += f"Pull {i+1}: Arm {action} -> Reward {reward}\n"
            
    return history_str

def make_raw_prompt(n_arms: int, history: List[Tuple[int, float]], template_type: str):
    """Make a raw prompt for a Multi-Armed Bandit problem in OpenAI chat format.
    
    Args:
        n_arms: Number of arms
        history: List of (action, reward) tuples
        template_type: Template type
        
    Returns:
        List of message dictionaries in OpenAI chat format
    """
    history_str = format_history(n_arms, history, summarized_history=True)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes Multi-Armed Bandit problems."
        },
        {
            "role": "user",
            "content": f"In a {n_arms}-armed bandit problem, here are the results of previous arm pulls:\n\n{history_str}\nWhich arm should be pulled next? Show your reasoning in <think> </think> tags and your final answer in <answer> </answer> tags."
        }
    ]
    return messages


# ================= Contextual Bandit (MovieLens) wrapper =================

class MovieLensCBEnv(gym.Env):
    """
    Gymnasium wrapper for banditbench MovieLens contextual bandit.

    - Uses banditbench MovieLens for stepping with integer actions.
    - Uses MovieLensVerbal to generate user-facing prompts and action names.
    - Computes ucb_action via LinUCBAgent and provides ucb_reward meta-info
      analogous to the MAB example.

    Observation: dummy (Discrete(1)); the prompt is returned in `info['raw_prompt']`.
    Action: Discrete(num_arms) where each action indexes into MovieLens arms.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        task_name: str = "100k-ratings",
        n_arms: int = 5,
        max_steps: int = 50,
        rank_k: int = 5,
        mode: str = "train",
        seed: Optional[int] = None,
        linucb_alpha: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_arms = n_arms
        self.horizon = max_steps
        self.rank_k = rank_k
        self.mode = mode
        self.seed_value = seed

        # Core bandit and verbal helper share the same underlying bandit
        self.core_bandit = MovieLens(
            task_name=task_name,
            num_arms=n_arms,
            horizon=max_steps,
            rank_k=rank_k,
            mode=mode,
            seed=seed,
        )
        self.verbal = MovieLensVerbal(self.core_bandit, instruction_type='base')

        # LinUCB oracle/guide used to compute ucb_action
        self.linucb = LinUCBAgent(self.core_bandit, alpha=linucb_alpha)

        # Gym spaces
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(1)  # dummy; see info['raw_prompt']

        # State and textual interaction history
        self._state = None 
        self.steps = 0
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.last_info: Dict[str, Any] = {}
        self._interactions: List[Tuple[str, str, float]] = []  # (context_text, action_name, reward)
        self.decision_context_start: str = (
            "So far you have interacted {} times with the most recent following choices and rewards:\n"
        )

    # ---- helpers ----
    @property
    def current_state(self):
        return self._state

    @property
    def action_names(self) -> List[str]:
        return self.verbal.get_actions_text()[: self.n_arms]

    def _best_action_and_value(self, state) -> Tuple[int, float, float]:
        """Return (best_action, best_value, worst_value) for the current state."""
        # Access the approximated rating matrix for this user
        user_idx = state.index
        row = self.core_bandit._approx_ratings_matrix[user_idx, : self.n_arms]  # noqa: SLF001
        best_action = int(np.argmax(row))
        best_value = float(row[best_action])
        worst_value = float(np.min(row))
        return best_action, best_value, worst_value

    def _history_snippet(self) -> str:
        if len(self._interactions) == 0:
            return ""
        lines = []
        for context_text, action, reward in self._interactions:
            lines.append(
                "\n" +
                f"Context: {context_text}\n" +
                f"Recommended movie {action + 1}\n" +
                f"Reward: {round(reward, 1)+0.0:g}\n"
            )
        return "".join(lines)

    def _make_raw_prompt(self, state: Any) :
        """Builds a raw prompt string identical to LLMCBAgentBase.act composition."""
        # Ensure feature_text available on state
        state.feature_text = self.verbal.verbalize_state(state)
        task_instruction = self.verbal.get_task_instruction()
        history_context = self.decision_context_start.format(len(self._interactions)) + self._history_snippet()
        query = self.verbal.get_query_prompt(state)
        return [
            {
                "role": "system",
                "content": task_instruction
            },
            {
                "role": "user",
                "content": history_context + query
            }
        ]

    # ---- gym API ----
    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        # Reset underlying components
        self.steps = 0
        self.counts[:] = 0
        self.core_bandit.reset(seed=seed if seed is not None else self.seed_value)
        self.linucb.reset()
        self._interactions = []

        # Sample initial state via verbal reset so feature_text and prompts align
        state, info = self.verbal.reset(seed=seed if seed is not None else self.seed_value)
        self._state = state

        # Compute oracle UCB action BEFORE the agent acts (used to score ucb_reward)
        ucb_action = int(self.linucb.select_action(self._state))

        # Regret is 0 at start; set max_regret based on current state's spread
        _, best_value, worst_value = self._best_action_and_value(self._state)
        raw_prompt = self._make_raw_prompt(self._state)
        self.last_info = {
            "regret": 0.0,
            "raw_prompt": raw_prompt,
            "ucb_action": ucb_action,
            "ucb_action_text": self.verbal.get_actions_text()[ucb_action],
            "ucb_reward": 0.0,
            "all_arms_pulled": False,
            "largest_count": 0,
            "max_regret": float(best_value - worst_value),
            "instruction": info.get("instruction") if isinstance(info, dict) else None,
        }

        return 0, self.last_info

    def step(self, action: int):
        # Validate action
        if not self.action_space.contains(action):
            return 0, 0.0, False, False, self.last_info

        # Execute in core bandit
        next_state, reward, done, _ = self.core_bandit.step(self._state, int(action))
        self.steps += 1
        self.counts[action] += 1

        # Update LinUCB with observed transition
        self.linucb.update(self._state, int(action), float(reward), info={})

        # Compute UCB-based reward against last precomputed ucb_action
        ucb_reward = float(action == self.last_info.get("ucb_action"))

        # Compute regret for this state
        _, best_value, worst_value = self._best_action_and_value(self._state)
        regret = float(best_value - float(reward))

        # Record interaction text for history (based on the pre-action state)
        prev_context_text = getattr(self._state, "feature_text", self.verbal.verbalize_state(self._state))
        self._interactions.append((prev_context_text, int(action), float(reward)))

        # Prepare next state's prompt and ucb action for the following step
        self._state = next_state
        next_ucb_action = int(self.linucb.select_action(self._state))
        raw_prompt = self._make_raw_prompt(self._state)

        # Episode termination
        done_flag = done or (self.steps >= self.horizon)

        all_arms_pulled = bool(np.all(self.counts > 0))
        largest_count = int(self.counts.max()) if self.counts.size else 0

        self.last_info = {
            "regret": regret,
            "raw_prompt": raw_prompt,
            "ucb_action": next_ucb_action,
            "ucb_action_text": self.verbal.get_actions_text()[next_ucb_action],
            "ucb_reward": ucb_reward,
            "all_arms_pulled": all_arms_pulled,
            "largest_count": largest_count,
            "max_regret": float(best_value - worst_value),
        }

        return 0, float(reward), done_flag, False, self.last_info
