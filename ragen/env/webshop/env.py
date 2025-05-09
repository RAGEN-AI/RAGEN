from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.webshop.config import WebShopEnvConfig
from webshop_minimal import WebAgentTextEnv, init_basedir
from typing import Optional, Union
from ragen.utils import all_seed
import random
import string
import uuid


# Define global constant for render instructions
RENDER_INSTRUCTIONS = [
    "We must buy a product within 10 actions. It doesn't have to match perfectly with description.",
    "Search term should not include details like size, color.",
    "Never search for more than 2 times.",
    "Do not be too strict about the description, it's more important to buy one that is close enough within action limit.",
    "Prioritize click a product in the current page over going to next page.",
    "Almost never click[next >] for more than 2 times."
    "Almost never click[< prev] unless you are sure the product is on one of the previous pages.",
    "If you have less than 3 actions left, just buy the first product you see in the current page.",
    "If an matching option exists, make sure to click[size] then click[color], one at a time, before click[buy now], but don't have to if only 1 action left, in that case you just click[buy now]. Never click description."
]


class WebShopEnv(BaseLanguageBasedEnv, WebAgentTextEnv):
    def __init__(self, config: Optional[WebShopEnvConfig] = None, **kwargs: any) -> None:
        """
        Adapter for WebAgentTextEnv to conform to the BaseLanguageBasedEnv interface.
        """
        self.config = config or WebShopEnvConfig()
        self.observation_mode = self.config.observation_mode
        self.file_path = self.config.file_path
        self.server = self.config.server
        self.filter_goals = self.config.filter_goals
        self.limit_goals = self.config.limit_goals
        self.num_products = self.config.num_products
        self.human_goals = self.config.human_goals
        self.show_attrs = self.config.show_attrs
        self.render_cache = None
        if self.config.dataset:
            init_basedir(self.config.dataset)

        BaseLanguageBasedEnv.__init__(self)
        WebAgentTextEnv.__init__(
            self,
            observation_mode=self.observation_mode,
            file_path=self.file_path,
            server=self.server,
            filter_goals=self.filter_goals,
            limit_goals=self.limit_goals,
            num_products=self.num_products,
            human_goals=self.human_goals,
            show_attrs=self.show_attrs,
            session_prefix=str(uuid.uuid4().hex), # we use a random session prefix to avoid collision
            **kwargs
        )

    def _get_permuted_index(self, idx, seed=42):
        """Map index to a deterministically permuted index in the same range.
        
        Args:
            idx: The original index
            seed: Random seed to ensure deterministic permutation
            
        Returns:
            int: The permuted index
        """
        # Create a cache key based on goals length and seed
        cache_key = f"perm_{len(self.server.goals)}_{seed}"
        
        # Create or retrieve the permutation map
        if not hasattr(self, cache_key):
            # Initialize with fixed seed
            rng = random.Random(seed)
            
            # Generate the full permutation
            indices = list(range(len(self.server.goals)))
            rng.shuffle(indices)
            
            # Store the permutation as an instance attribute
            setattr(self, cache_key, indices)
        
        # Look up the permuted index
        permutation = getattr(self, cache_key)
        return permutation[idx]

    def reset(self, seed=None, mode="train", session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None) -> any:
        """
        Reset the environment and return the initial observation.

        Args:
            session (str|int|None): The new session ID.
            instruction_text (str|None): Optional new instruction text.

        Returns:
            The initial observation.
        """
        if seed is None:
            # This is from within webshop_minimal. Need to reset with seed later.
            return None
        if mode == "test":
            goal_idx = seed % 500
        elif mode == "val":
            goal_idx = seed % 1000 + 500
        elif mode == "train":
            goal_idx = seed % (len(self.server.goals) - 1500) + 1500
        session = self._get_permuted_index(goal_idx) if session is None else session
        obs, _ = WebAgentTextEnv.reset(self, session=session, instruction_text=instruction_text)
        self.prepare_render_cache(WebAgentTextEnv.get_instruction_text(self))
        return obs

    def step(self, action):
        """
        Take an action in the environment and return the next observation, reward, done, and info.
        """
        action_is_valid = action in self.get_available_actions() or ("search[<content>]" in self.get_available_actions() and action.startswith('search[') and action.endswith(']'))
        last_observation = self.observation
        state, reward, done, info = WebAgentTextEnv.step(self, action)
        self.prepare_render_cache(self.observation)
            
        info = (info or {}).copy()
        info.update({
            "reward": reward,
            "action_is_effective": self.observation != last_observation,
            "action_is_valid": action_is_valid,
            "success": 1 if reward == 1 else 0,
            "success_purchase": 1 if done else 0,
            "success_find": 1 if reward == 1 else 0,
            "end_of_page": 1 if tuple(self.get_available_actions()) == ('click[back to search]', 'click[< prev]') else 0,
        })
        return self.observation, reward, done, info

    def render(self, mode=None):
        """
        Render the environment.
        """
        return self.render_cache

    def close(self):
        """
        Close the environment.
        """
        WebAgentTextEnv.close(self)

    def prepare_render_cache(self, observation: str):
        """
        Prepare the render cache for the environment.
        """
        available_actions = self.get_available_actions()
        self.render_cache = observation + "."
        self.render_cache += "\n".join(RENDER_INSTRUCTIONS)
        self.render_cache += "\n You must choose from these actions:" + ", ".join(available_actions) + "."
        

    def get_available_actions(self):
        """
        Parse the available actions in the environment to a list of strings.
        """
        orig_available_actions = WebAgentTextEnv.get_available_actions(self)
        available_actions = []

        if orig_available_actions['has_search_bar']:
            available_actions.append('search[<content>]')

        for clickable in orig_available_actions['clickables']:
            if clickable != 'search':
                available_actions.append(f'click[{clickable}]')
        # TODO: we may need to purge the case when available_actions == ['click[back to search]', 'click[< prev]', 'click[next >]']
        is_end_of_page = tuple(available_actions) == ('click[back to search]', 'click[< prev]', 'click[next >]')
        if is_end_of_page:
            available_actions.remove('click[next >]')
        return available_actions

if __name__ == '__main__':
    env = WebShopEnv()
    print(env.reset())
    while True:
        print(env.observation)
        print(env.server.user_sessions[env.session]['goal']['asin'])
        print(f"Available actions: {env.get_available_actions()}")
        action = input("Enter action: ")
        if action == 'q':
            break
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    env.close()
