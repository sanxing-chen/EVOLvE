import os
from typing import List, Optional

from banditbench.tasks.types import State
from banditbench.utils import dedent
from banditbench.tasks.scenario import CBScenario

class ScenarioUtil:
    """
    Inheriting from this class unifies the subclass' default __init__ method
    """

    def __init__(self, action_names, num_actions: int,
                 seed: Optional[int] = None):
        super().__init__(num_actions=num_actions, action_names=action_names, action_unit=self.action_unit,
                         base_description=self.base_description, detailed_description=self.detailed_description,
                         query_prompt=self.query_prompt, seed=seed)


class MovieLensScenario(ScenarioUtil, CBScenario):
    name = 'movie'
    action_unit = 'movie'
    reward_unit = 'User returned a rating of'
    max_reward = '5'

    base_description = dedent(
        """You are an AI movie recommendation assistant for a streaming platform powered by a bandit algorithm that offers a wide variety of films.
          There are {} unique movies you can recommend, named {}
          When a user visits the streaming platform, you assess their demographic description to choose a movie to suggest.
          You aim to match the user with movies they are most likely to watch and enjoy.
          Each time a user watches a recommended movie, you adjust your recommendation algorithms to better predict and meet future user preferences.
        """
    )

    detailed_description = dedent(
        """You are an AI movie recommendation assistant for a streaming platform powered by a bandit algorithm that offers a wide variety of films.
          There are {} unique movies you can recommend, named {}
          When a user visits the streaming platform, you assess their demographic description to choose a movie to suggest.
          You aim to match the user with movies they are most likely to watch and enjoy.
          Each time a user watches a recommended movie, you adjust your recommendation algorithms to better predict and meet future user preferences.
          
          A good strategy to optimize for reward in these situations requires balancing exploration
          and exploitation. You need to explore to try out different movies and find those
          with high rewards, but you also have to exploit the information that you have to
          accumulate rewards.
        """
    )

    query_prompt = ("\n\nYou have a new user: \nContext: {feature}\nWhich movie (id) should be recommended next? Show your reasoning in <think> </think> tags and your final answer in <answer> Movie n </answer> tags.")

    fewshot_prompt = ('Here are some examples of optimal actions for different users.'
                      ' Use them as hints to help you come up with better actions.\n')

    action_names: List[str]

    def get_instruction(self, version="base"):
        """We add the few-shot examples in here"""
        formatted_actions = '\n' + '\n'.join(f'{i+1}. {movie}' for i, movie in enumerate(self.action_names)) + '\n'
        if version == 'base':
            prompt = self.base_description.format(
                len(self.action_names), formatted_actions
            )
        elif version == 'detailed':
            prompt = self.detailed_description.format(
                len(self.action_names), formatted_actions
            )
        else:
            raise ValueError(f'Unknown description version {version}')

        return prompt

    def get_query_prompt(self, state: State, side_info: Optional[str] = None):
        assert type(
            state.feature_text) is str, "Verbal bandit produces string as feature, make sure you are using the right bandit class"
        prompt = self.query_prompt.format(feature=state.feature_text)
        if side_info is not None:
            prompt += f"Side Information for decision making:\n{side_info}"

        return prompt
