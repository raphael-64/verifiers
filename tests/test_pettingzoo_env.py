from __future__ import annotations

import os
import re

import pytest
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from verifiers.envs.experimental.pettingzoo_env import PettingZooAECEnv

pettingzoo = pytest.importorskip("pettingzoo")
from pettingzoo.test.example_envs import (
    generated_agents_env_action_mask_info_v0,
    generated_agents_env_action_mask_obs_v0,
    generated_agents_env_v0,
)


class _MockChatCompletions:
    async def create(self, *, model: str, messages: list[dict[str, str]], **kwargs):
        message = ChatCompletionMessage.model_construct(
            role="assistant", content="0", tool_calls=None
        )
        choice = Choice.model_construct(
            index=0, message=message, finish_reason="stop", logprobs=None
        )
        return ChatCompletion.model_construct(
            id="mock-chatcmpl",
            choices=[choice],
            created=0,
            model=model,
            object="chat.completion",
            usage=None,
        )


class _MockChat:
    def __init__(self):
        self.completions = _MockChatCompletions()


class MockAsyncOpenAI:
    def __init__(self):
        self.chat = _MockChat()
        self.base_url = "mock://local"


class _MockChatCompletionsInvalid:
    async def create(self, *, model: str, messages: list[dict[str, str]], **kwargs):
        message = ChatCompletionMessage.model_construct(
            role="assistant", content="999999", tool_calls=None
        )
        choice = Choice.model_construct(
            index=0, message=message, finish_reason="stop", logprobs=None
        )
        return ChatCompletion.model_construct(
            id="mock-chatcmpl",
            choices=[choice],
            created=0,
            model=model,
            object="chat.completion",
            usage=None,
        )


class _MockChatInvalid:
    def __init__(self):
        self.completions = _MockChatCompletionsInvalid()


class MockAsyncOpenAIInvalid:
    def __init__(self):
        self.chat = _MockChatInvalid()
        self.base_url = "mock://local"

def parse_action(txt: str) -> int:
    m = re.search(r"[-+]?\d+", txt)
    if not m:
        raise ValueError(f"No int in: {txt!r}")
    return int(m.group(0))

def test_basic_aec_rollout_mock():
    client = MockAsyncOpenAI()
    env = PettingZooAECEnv(
        env_cls=generated_agents_env_v0.env,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=10,
    )

    outputs = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "pz_done"]
    )
    st = outputs["outputs"][0]
    steps = st.get("trajectory", [])

    assert len(steps) > 0
    assert st.get("is_completed") is True
    if st.get("pz_done") is False:
        assert len(steps) == 10

    first_prompt = st["prompt"]
    assert isinstance(first_prompt, list)
    assert "Agent:" in (first_prompt[-1].get("content") or "")

    assert any("agent_id" in (s.get("extras") or {}) for s in steps)
    assert any("pz_rewards" in (s.get("extras") or {}) for s in steps)


class _TinyAECEnv:
    def __init__(self, max_cycles: int = 2):
        self.max_cycles = int(max_cycles)
        self.agents = ["a0", "a1"]
        self.agent_selection = "a0"
        self.rewards = {}
        self._cycle = 0
        self._terminated = {a: False for a in self.agents}
        self._truncated = {a: False for a in self.agents}

    def reset(self, seed=None, options=None):
        self.agent_selection = "a0"
        self.rewards = {a: 0.0 for a in self.agents}
        self._cycle = 0
        self._terminated = {a: False for a in self.agents}
        self._truncated = {a: False for a in self.agents}

    def last(self):
        obs = f"obs:{self.agent_selection}:{self._cycle}"
        reward = self.rewards.get(self.agent_selection, 0.0)
        term = self._terminated[self.agent_selection]
        trunc = self._truncated[self.agent_selection]
        info = {}
        return obs, reward, term, trunc, info

    def step(self, action):
        if self._terminated[self.agent_selection] or self._truncated[
            self.agent_selection
        ]:
            self.agent_selection = (
                "a1" if self.agent_selection == "a0" else "a0"
            )
            if all(self._terminated.values()):
                self.agents = []
            return
        self.rewards[self.agent_selection] = 1.0
        if self.agent_selection == "a1":
            self._cycle += 1
            if self._cycle >= self.max_cycles:
                for a in self.agents:
                    self._terminated[a] = True
                self.agents = []
        self.agent_selection = "a1" if self.agent_selection == "a0" else "a0"

    def close(self):
        pass


@pytest.mark.integration
def test_basic_aec_rollout_live():
    if os.getenv("RUN_LIVE_OPENAI") != "1":
        pytest.skip("Set RUN_LIVE_OPENAI=1 to run live OpenAI integration test.")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = AsyncOpenAI()
    env = PettingZooAECEnv(
        env_cls=generated_agents_env_action_mask_obs_v0.raw_env,
        env_kwargs={"max_cycles": 2},
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=10,
    )

    outputs = env.evaluate_sync(
        client=client, model=model, state_columns=["trajectory", "pz_done"]
    )
    st = outputs["outputs"][0]
    steps = st.get("trajectory", [])

    assert len(steps) > 0
    assert st.get("is_completed") is True
    if st.get("pz_done") is False:
        assert len(steps) == 10
    assert "Action space:" in str(st["prompt"])
    assert "Action mask:" in str(st["prompt"])


def test_aec_rollout_terminates():
    client = MockAsyncOpenAI()
    env = PettingZooAECEnv(
        env_cls=_TinyAECEnv,
        env_kwargs={"max_cycles": 2},
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=20,
    )
    outputs = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "pz_done"]
    )
    st = outputs["outputs"][0]
    steps = st.get("trajectory", [])
    assert len(steps) > 0
    assert st.get("pz_done") is True
    assert st.get("is_completed") is True


def test_action_parse_error_ends_episode():
    def bad_parser(_txt: str) -> int:
        raise ValueError("no action")

    client = MockAsyncOpenAI()
    env = PettingZooAECEnv(
        env_cls=_TinyAECEnv,
        action_parser=bad_parser,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=5,
    )
    res = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "pz_done"]
    )
    st = res["outputs"][0]
    steps = st.get("trajectory", [])
    assert st.get("pz_done") is True
    last_prompt = steps[-1]["prompt"]
    assert "Action Parsing Error" in str(last_prompt)


def test_max_episode_steps_limits_turns():
    class NoTermEnv(_TinyAECEnv):
        def step(self, action):
            self.rewards[self.agent_selection] = 0.0
            self.agent_selection = (
                "a1" if self.agent_selection == "a0" else "a0"
            )

    client = MockAsyncOpenAI()
    env = PettingZooAECEnv(
        env_cls=NoTermEnv,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=3,
    )
    res = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "pz_done"]
    )
    st = res["outputs"][0]
    steps = st.get("trajectory", [])
    assert len(steps) == 3
    assert st.get("pz_done") is False
    assert st.get("is_completed") is True


def test_dataset_generation_chat_mode():
    env = PettingZooAECEnv(
        env_cls=_TinyAECEnv,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=4,
        num_eval_episodes=2,
    )
    assert env.dataset is not None
    assert env.eval_dataset is not None
    assert len(env.dataset) == 4
    assert len(env.eval_dataset) == 2
    assert "question" in env.dataset.column_names


def test_dataset_generation_completion_mode():
    env = PettingZooAECEnv(
        env_cls=_TinyAECEnv,
        action_parser=parse_action,
        message_type="completion",
        num_train_episodes=3,
        num_eval_episodes=1,
    )
    assert env.dataset is not None
    assert len(env.dataset) == 3
    assert "prompt" in env.dataset.column_names
    assert isinstance(env.dataset[0]["prompt"], str)


def test_env_kwargs_passed_to_env():
    env = PettingZooAECEnv(
        env_cls=_TinyAECEnv,
        env_kwargs={"max_cycles": 1},
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=10,
    )
    res = env.evaluate_sync(
        client=MockAsyncOpenAI(), model="mock", state_columns=["pz_done"]
    )
    st = res["outputs"][0]
    assert st.get("pz_done") is True


def test_action_mask_in_prompt():
    env = PettingZooAECEnv(
        env_cls=generated_agents_env_action_mask_obs_v0.raw_env,
        env_kwargs={"max_cycles": 1},
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=5,
    )
    res = env.evaluate_sync(client=MockAsyncOpenAI(), model="mock")
    st = res["outputs"][0]
    first_prompt = st["prompt"]
    assert "Action mask:" in str(first_prompt)


def test_action_mask_info_in_prompt():
    env = PettingZooAECEnv(
        env_cls=generated_agents_env_action_mask_info_v0.raw_env,
        env_kwargs={"max_cycles": 1},
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=5,
    )
    res = env.evaluate_sync(client=MockAsyncOpenAI(), model="mock")
    st = res["outputs"][0]
    first_prompt = st["prompt"]
    assert "Action mask:" in str(first_prompt)


def test_invalid_action_ends_episode():
    env = PettingZooAECEnv(
        env_cls=generated_agents_env_action_mask_obs_v0.raw_env,
        env_kwargs={"max_cycles": 1},
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=5,
    )
    res = env.evaluate_sync(
        client=MockAsyncOpenAIInvalid(),
        model="mock",
        state_columns=["pz_done", "trajectory"],
    )
    st = res["outputs"][0]
    assert st.get("pz_done") is True
    last_prompt = st["trajectory"][-1]["prompt"]
    assert "Invalid action" in str(last_prompt)


def test_skips_terminated_agent_on_reset():
    class _DeadFirstAECEnv(_TinyAECEnv):
        def reset(self, seed=None, options=None):
            super().reset(seed=seed, options=options)
            self._terminated["a0"] = True
            self.agent_selection = "a0"

        def step(self, action):
            if self._terminated[self.agent_selection]:
                self.agent_selection = "a1"
                return
            super().step(action)

    env = PettingZooAECEnv(
        env_cls=_DeadFirstAECEnv,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
        max_episode_steps=5,
    )
    res = env.evaluate_sync(client=MockAsyncOpenAI(), model="mock")
    st = res["outputs"][0]
    completion = st["completion"]
    assert "Agent: a1" in str(completion)
