"""Compatibility adapter with an exact action bound and terminal observation flush."""

from __future__ import annotations

import logging
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .explorer import Explorer
from .state import encode


class Stage1Agent(Agent):
    MAX_ACTIONS = 1000
    MODE = "graph"

    def __init__(
        self,
        *args: Any,
        seed: int = 0,
        budget: int = 1000,
        emit: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.MAX_ACTIONS = budget
        self.action_counter = 0
        self.explorer = Explorer(self.MODE, seed, budget, emit)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        self.explorer.observe(encode(latest_frame))
        if latest_frame.state is GameState.WIN:
            self.explorer.stop_reason = "win"
        elif self.action_counter >= self.MAX_ACTIONS:
            self.explorer.stop_reason = "budget"
        return self.explorer.stop_reason is not None

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self.explorer.observe(encode(latest_frame))
        action = self.explorer.choose()
        if action is None:
            raise StopIteration(self.explorer.stop_reason)
        # The enum is shared across threads. Do not attach mutable data to it.
        # Our take_action uses the immutable pending Action instead.
        return GameAction.from_id(action.id)

    def do_action_request(self, action: GameAction) -> FrameData:
        pending = self.explorer.pending
        if pending is None or pending[1].id != action.value:
            raise RuntimeError("Action does not match the pending controller decision")
        raw = self.arc_env.step(
            action, data=pending[1].data, reasoning={"agent": f"stage1{self.MODE}"}
        )
        return self._convert_raw_frame_data(raw)

    def main(self) -> None:
        self.timer = time.time()
        try:
            latest = self._convert_raw_frame_data(self.arc_env.observation_space)
            self.frames = [latest]
            while not self.is_done(self.frames, latest):
                try:
                    action = self.choose_action(self.frames, latest)
                except StopIteration:
                    break
                self.action_counter += 1
                result = self.take_action(action)
                if result is None:
                    raise RuntimeError("Environment returned no valid observation")
                latest = result
                self.append_frame(latest)
                # Keep bounded observation history; full unique frames live in the graph.
                self.frames = self.frames[-2:]
                self.explorer.observe(encode(latest))
        except Exception:
            self.explorer.stop_reason = "error"
            logging.getLogger(__name__).exception(
                "Stage 1 game failed: %s", self.game_id
            )
            raise
        finally:
            self.cleanup()


class Stage1Random(Stage1Agent):
    MODE = "random"


class Stage1NoRepeat(Stage1Agent):
    MODE = "norepeat"


class Stage1Graph(Stage1Agent):
    MODE = "graph"
