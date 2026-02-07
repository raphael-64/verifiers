from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, cast

from datasets import Dataset

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import MessageType, State


class AECEnv(Protocol):
    """
    Protocol for PettingZoo AEC environments compatible with PettingZooAECEnv.

    Required methods/attributes:
        reset(seed: int) -> None | (obs, info)
            Reset the environment. Must accept an optional `seed` keyword argument.
            Many PettingZoo envs return None, but some return (obs, info).

        last() -> (obs, reward, terminated, truncated, info)
            Get data for the current agent.

        step(action) -> None
            Take a step for the current agent (action can be None if agent done).

        agent_selection: str
            The currently active agent id.

        agents: list[str]
            The current living agents list. Empty when episode is over.

        rewards: dict[str, float]
            Reward values for the most recent step.
    """

    reset: Callable[..., Any]
    last: Callable[..., tuple[Any, float, bool, bool, dict[str, Any]]]
    step: Callable[..., Any]
    agent_selection: str
    agents: list[str]
    rewards: dict[str, float]


ResetOut: TypeAlias = Any | tuple[Any, dict[str, Any]]


def normalize_reset(out: ResetOut) -> tuple[Any | None, dict[str, Any]]:
    if isinstance(out, tuple) and len(out) == 2:
        return cast(tuple[Any, dict[str, Any]], out)
    return out, {}


def sum_step_rewards(state: State) -> float:
    return float(
        sum(
            float(step.get("reward", 0.0) or 0.0)
            for step in state.get("trajectory", [])
        )
    )


class EpisodicSumRubric(Rubric):
    def __init__(self, weight: float = 1.0, **kwargs: Any):
        super().__init__(funcs=[sum_step_rewards], weights=[weight], **kwargs)


class PettingZooAECEnv(vf.MultiTurnEnv):
    """Runner for PettingZoo AEC environments (turn-based multi-agent)."""

    def __init__(
        self,
        env_cls: Callable[..., AECEnv],
        env_kwargs: dict[str, Any] | None = None,
        action_parser: Callable[[str], Any] | None = None,
        obs_to_text: Callable[[Any], str] | None = None,
        include_action_mask: bool = True,
        num_train_episodes: int = 1000,
        num_eval_episodes: int = 20,
        max_episode_steps: int | None = None,
        seed: int = 0,
        # global
        system_prompt: str | None = None,
        few_shot: list[dict[str, Any]] | None = None,
        parser: vf.Parser | None = None,
        rubric: Rubric | None = None,
        message_type: MessageType = "chat",
    ):
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs or {})
        self.action_parser = action_parser or (lambda x: x)
        self.obs_to_text_fn = obs_to_text
        self.include_action_mask = include_action_mask
        self.num_train_episodes = num_train_episodes
        self.num_eval_episodes = num_eval_episodes
        self.seed = seed
        self.message_type = message_type

        dataset, eval_dataset = self.aec_to_hf()

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric or EpisodicSumRubric(),
            message_type=message_type,
            max_turns=max_episode_steps or 1000,
            system_prompt=system_prompt,
            few_shot=few_shot,
            parser=parser,
        )

    def _create_env(self) -> AECEnv:
        try:
            return cast(AECEnv, self.env_cls(**self.env_kwargs))
        except TypeError as e:
            if not self.env_kwargs:
                return cast(AECEnv, self.env_cls())
            env = cast(AECEnv, self.env_cls())
            unused: list[str] = []
            for key, value in self.env_kwargs.items():
                if hasattr(env, key):
                    setattr(env, key, value)
                else:
                    unused.append(key)
            if unused:
                raise TypeError(
                    f"{self.env_cls} does not accept env_kwargs: {unused}"
                ) from e
            return env

    def _reset_env(self, env: AECEnv, seed: int) -> tuple[Any | None, dict[str, Any]]:
        try:
            return normalize_reset(env.reset(seed=seed))
        except TypeError:
            return normalize_reset(env.reset())

    def aec_to_hf(self) -> tuple[Dataset, Dataset | None]:
        train_rows = []
        eval_rows = []
        total = self.num_train_episodes + self.num_eval_episodes
        env = self._create_env()

        try:
            for i in range(total):
                self._reset_env(env, seed=self.seed + i)
                if not getattr(env, "agents", None):
                    raise RuntimeError("PettingZoo env has no agents after reset.")
                agent_id = env.agent_selection
                obs, _reward, _term, _trunc, info = env.last()
                action_space = self._get_action_space(env, agent_id)
                question = self.format_observation(
                    agent_id, obs, info, action_space=action_space
                )
                if self.message_type == "completion":
                    row = {"prompt": question, "answer": str(self.seed + i)}
                else:
                    row = {"question": question, "answer": str(self.seed + i)}
                if i < self.num_train_episodes:
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
        finally:
            close_fn = getattr(env, "close", None)
            if close_fn is not None:
                close_fn()

        dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
        return dataset, eval_dataset

    def obs_to_text(self, obs: Any) -> str:
        """Convert observation to text. Override in subclass for custom formatting."""
        if self.obs_to_text_fn:
            return self.obs_to_text_fn(obs)
        return str(obs)

    def _extract_action_mask(self, obs: Any, info: dict[str, Any]) -> Any | None:
        if isinstance(info, dict) and "action_mask" in info:
            return info.get("action_mask")
        if isinstance(obs, dict) and "action_mask" in obs:
            return obs.get("action_mask")
        return None

    def _get_action_space(self, env: AECEnv, agent_id: str) -> Any | None:
        action_space = getattr(env, "action_space", None)
        if callable(action_space):
            try:
                return action_space(agent_id)
            except Exception:
                return None
        return None

    def _format_action_space(self, space: Any | None) -> str | None:
        if space is None:
            return None
        try:
            return str(space)
        except Exception:
            return None

    def _validate_action(self, space: Any | None, action: Any) -> bool:
        if space is None:
            return True
        contains = getattr(space, "contains", None)
        if callable(contains):
            try:
                return bool(contains(action))
            except Exception:
                return False
        return True

    def format_observation(
        self, agent_id: str, obs: Any, info: dict[str, Any], action_space: Any | None
    ) -> str:
        obs_text = self.obs_to_text(obs)
        lines = [f"Agent: {agent_id}", f"Observation: {obs_text}"]
        action_space_text = self._format_action_space(action_space)
        if action_space_text:
            lines.append(f"Action space: {action_space_text}")
        if self.include_action_mask:
            mask = self._extract_action_mask(obs, info)
            if mask is not None:
                lines.append(f"Action mask: {mask}")
        return "\n".join(lines)

    def wrap_response(self, text: str) -> vf.Messages:
        if self.message_type == "chat":
            return cast(vf.Messages, [{"role": "user", "content": text}])
        return text

    def _get_action_text(self, messages: vf.Messages, state: State) -> str:
        raw_text = self.parser.parse_answer(messages)
        if raw_text is None:
            last_completion = state["trajectory"][-1]["completion"]
            if isinstance(last_completion, list) and last_completion:
                raw_text = str(last_completion[-1].get("content", ""))
            else:
                raw_text = str(last_completion)
        return str(raw_text)

    async def env_response(
        self, messages: vf.Messages, state: State, **kwargs: Any
    ) -> vf.Messages:
        if "pz_env" not in state:
            env = self._create_env()
            seed = int(state["answer"])
            self._reset_env(env, seed=seed)
            state["pz_env"] = env
            state["pz_done"] = False
        else:
            env = state["pz_env"]

        if not getattr(env, "agents", None):
            state["pz_done"] = True
            return self.wrap_response("Episode already ended.")

        agent_id = env.agent_selection
        obs, _reward, terminated, truncated, info = env.last()

        # If the current agent is already done, advance with None (ignore action).
        while terminated or truncated:
            env.step(None)
            if not getattr(env, "agents", None):
                state["pz_done"] = True
                return self.wrap_response("Episode already ended.")
            agent_id = env.agent_selection
            obs, _reward, terminated, truncated, info = env.last()

        action_text = self._get_action_text(messages, state)
        try:
            action = self.action_parser(action_text)
        except Exception as e:
            state["pz_done"] = True
            state["trajectory"][-1]["reward"] = 0.0
            err_text = f"Action Parsing Error: {e}"
            return self.wrap_response(err_text)

        action_space = self._get_action_space(env, agent_id)
        if not self._validate_action(action_space, action):
            state["pz_done"] = True
            state["trajectory"][-1]["reward"] = 0.0
            err_text = "Invalid action for agent action space."
            return self.wrap_response(err_text)
        env.step(action)

        reward = float(getattr(env, "rewards", {}).get(agent_id, 0.0))
        state["trajectory"][-1]["reward"] = reward
        extras = state["trajectory"][-1]["extras"]
        extras["agent_id"] = agent_id
        extras["pz_info"] = info
        extras["pz_action"] = action
        extras["pz_rewards"] = dict(getattr(env, "rewards", {}) or {})

        # Advance to the next active agent to build the next prompt.
        while True:
            if not getattr(env, "agents", None):
                state["pz_done"] = True
                return self.wrap_response("Episode already ended.")
            next_agent = env.agent_selection
            obs, _reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            next_action_space = self._get_action_space(env, next_agent)
            obs_text = self.format_observation(
                next_agent, obs, info, next_action_space
            )
            return self.wrap_response(obs_text)

    @vf.stop
    async def is_done(self, state: State) -> bool:
        return state.get("pz_done", False)

    @vf.cleanup
    async def cleanup_env(self, state: State) -> None:
        env = state.pop("pz_env", None)
        if env is not None:
            close_fn = getattr(env, "close", None)
            if close_fn is not None:
                close_fn()
