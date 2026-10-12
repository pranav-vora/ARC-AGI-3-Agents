"""Swarm orchestration contracts against the current Arcade toolkit API."""

from unittest.mock import Mock, patch

import pytest
from arc_agi import OperationMode
from arc_agi.scorecard import EnvironmentScorecard

from agents.swarm import Swarm
from agents.templates.random_agent import Random


@pytest.fixture(autouse=True)
def arcade():
    """Unit tests must never discover/download real games or open remote cards."""
    with patch("agents.swarm.Arcade") as factory:
        instance = factory.return_value
        instance.operation_mode = OperationMode.OFFLINE
        instance.open_scorecard.return_value = "test-card-123"
        instance.close_scorecard.return_value = EnvironmentScorecard(
            card_id="test-card-123"
        )
        yield instance


@pytest.mark.unit
def test_swarm_initialization(arcade, monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "test-api-key")
    swarm = Swarm(
        agent="random", ROOT_URL="https://example.com", games=["game1", "game2"]
    )
    assert swarm.agent_name == "random"
    assert swarm.ROOT_URL == "https://example.com"
    assert swarm.GAMES == ["game1", "game2"]
    assert swarm.agent_class is Random
    assert swarm.threads == swarm.agents == swarm.cleanup_threads == []
    assert swarm.headers == {"X-API-Key": "test-api-key", "Accept": "application/json"}
    assert swarm._arc is arcade


@pytest.mark.unit
@pytest.mark.parametrize("tags", [None, [], ["experiment1", "version2"]])
def test_scorecard_open_delegates_tags_without_mutating_caller(arcade, tags):
    kwargs = {} if tags is None else {"tags": tags}
    original = None if tags is None else tags.copy()
    swarm = Swarm(agent="random", ROOT_URL="", games=["game1"], **kwargs)
    assert swarm.open_scorecard() == "test-card-123"
    arcade.open_scorecard.assert_called_once_with(
        tags=(tags or []) + ["agent", "random"]
    )
    assert tags == original


@pytest.mark.unit
def test_scorecard_close_delegates_and_clears_active_card(arcade):
    swarm = Swarm(agent="random", ROOT_URL="", games=["game1"])
    swarm.card_id = "test-card-123"
    result = swarm.close_scorecard("test-card-123")
    assert result is arcade.close_scorecard.return_value
    arcade.close_scorecard.assert_called_once_with("test-card-123")
    assert swarm.card_id is None
    arcade.close_scorecard.return_value = None
    assert swarm.close_scorecard("missing") is None


@pytest.mark.unit
def test_scorecard_errors_propagate(arcade):
    swarm = Swarm(agent="random", ROOT_URL="", games=[])
    arcade.open_scorecard.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        swarm.open_scorecard()
    arcade.close_scorecard.side_effect = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        swarm.close_scorecard("test-card")


@pytest.mark.unit
def test_agent_creation_and_thread_lifecycle(arcade):
    swarm = Swarm(agent="random", ROOT_URL="", games=["game1", "game2", "game3"])
    agent_factory = Mock()
    agents = [Mock() for _ in swarm.GAMES]
    agent_factory.side_effect = agents
    swarm.agent_class = agent_factory
    threads = [Mock() for _ in swarm.GAMES]
    with patch("agents.swarm.Thread", side_effect=threads) as thread_factory:
        result = swarm.main()
    assert result is arcade.close_scorecard.return_value
    assert swarm.agents == agents
    assert thread_factory.call_count == 3
    assert arcade.make.call_count == 3
    for index, (game, agent, thread) in enumerate(zip(swarm.GAMES, agents, threads)):
        assert arcade.make.call_args_list[index].args == (game,)
        assert arcade.make.call_args_list[index].kwargs == {
            "scorecard_id": "test-card-123"
        }
        assert (
            agent_factory.call_args_list[index].kwargs["arc_env"]
            is arcade.make.return_value
        )
        assert thread_factory.call_args_list[index].kwargs == {
            "target": agent.main,
            "daemon": True,
        }
        thread.start.assert_called_once()
        thread.join.assert_called_once()
        agent.cleanup.assert_called_once_with(result)


@pytest.mark.unit
def test_cleanup_with_and_without_scorecard(arcade):
    swarm = Swarm(agent="random", ROOT_URL="", games=[])
    agents = [Mock(), Mock()]
    swarm.agents = agents
    card = EnvironmentScorecard(card_id="test-card")
    swarm.cleanup(card)
    for agent in agents:
        agent.cleanup.assert_called_once_with(card)
        agent.reset_mock()
    swarm.cleanup()
    for agent in agents:
        agent.cleanup.assert_called_once_with(None)


@pytest.mark.unit
def test_optional_legacy_session_cleanup(arcade):
    swarm = Swarm(agent="random", ROOT_URL="", games=[])
    swarm._session = Mock()
    swarm.cleanup()
    swarm._session.close.assert_called_once()


@pytest.mark.unit
def test_playback_tags(arcade):
    from agents import AVAILABLE_AGENTS, Playback

    name = "game.random.80.example-guid.recording.jsonl"
    with patch.dict(AVAILABLE_AGENTS, {name: Playback}):
        swarm = Swarm(agent=name, ROOT_URL="", games=["game"])
    assert swarm.tags == ["playback", "example-guid"]
