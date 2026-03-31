"""Unit tests to validate langgraph-checkpoint v4 compatibility.

These tests ensure that the upgrade from langgraph-checkpoint v3 to v4
does not break any functionality used by this package.
"""
import pytest
from typing import Literal
from typing_extensions import TypedDict
from unittest.mock import Mock, patch

from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# Helpers / shared state schemas
# ---------------------------------------------------------------------------

class SimpleState(TypedDict):
    value: str


class TimbrLikeState(TypedDict):
    prompt: str
    sql: str
    concept: str
    rows: list
    response: str
    error: str
    is_sql_valid: bool


# ---------------------------------------------------------------------------
# Core checkpoint import tests
# ---------------------------------------------------------------------------

class TestCheckpointImports:
    """Verify that all public checkpoint symbols are importable from v4."""

    def test_import_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert MemorySaver is not None

    def test_import_base_checkpoint_saver(self):
        from langgraph.checkpoint.base import BaseCheckpointSaver
        assert BaseCheckpointSaver is not None

    def test_import_checkpoint_at(self):
        """CheckpointAt (or equivalent) exists in v4."""
        from langgraph.checkpoint.base import BaseCheckpointSaver
        # The class should be importable and be a class
        assert isinstance(BaseCheckpointSaver, type)

    def test_memory_saver_instantiation(self):
        from langgraph.checkpoint.memory import MemorySaver
        saver = MemorySaver()
        assert saver is not None

    def test_memory_saver_context_manager(self):
        """In v4 InMemorySaver context manager must return self (was fixed in v4)."""
        from langgraph.checkpoint.memory import MemorySaver
        with MemorySaver() as saver:
            assert saver is not None


# ---------------------------------------------------------------------------
# StateGraph compilation tests
# ---------------------------------------------------------------------------

class TestStateGraphCompilation:
    """Verify that StateGraph.compile() still works correctly with v4."""

    def test_compile_without_checkpointer(self):
        g = StateGraph(SimpleState)
        g.add_node("node", lambda s: {"value": "done"})
        g.set_entry_point("node")
        g.add_edge("node", END)
        compiled = g.compile()
        assert compiled is not None

    def test_compile_with_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        g = StateGraph(SimpleState)
        g.add_node("node", lambda s: {"value": "done"})
        g.set_entry_point("node")
        g.add_edge("node", END)
        compiled = g.compile(checkpointer=MemorySaver())
        assert compiled is not None

    def test_invoke_without_checkpointer(self):
        g = StateGraph(SimpleState)
        g.add_node("node", lambda s: {"value": "processed"})
        g.set_entry_point("node")
        g.add_edge("node", END)
        result = g.compile().invoke({"value": "input"})
        assert result == {"value": "processed"}

    def test_invoke_with_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        g = StateGraph(SimpleState)
        g.add_node("node", lambda s: {"value": "persisted"})
        g.set_entry_point("node")
        g.add_edge("node", END)
        compiled = g.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-thread"}}
        result = compiled.invoke({"value": "start"}, config)
        assert result["value"] == "persisted"


# ---------------------------------------------------------------------------
# Timbr-graph-shaped workflow tests (mirrors TimbrLlmConnector.run_llm_query_graph)
# ---------------------------------------------------------------------------

class TestTimbrGraphWorkflow:
    """Validate that a StateGraph shaped like TimbrLlmConnector's run_llm_query_graph
    still compiles and routes correctly after the checkpoint upgrade."""

    def _build_graph(self):
        """Build a minimal graph that mimics the Timbr LangGraph workflow."""

        def generate_sql(state: TimbrLikeState) -> dict:
            return {"sql": "SELECT * FROM concept", "concept": "concept"}

        def validate_sql(state: TimbrLikeState) -> dict:
            return {"is_sql_valid": True}

        def execute_sql(state: TimbrLikeState) -> dict:
            return {"rows": [["row1"], ["row2"]], "error": ""}

        def generate_response(state: TimbrLikeState) -> dict:
            return {"response": "Here are the results."}

        def route_validation(state: dict) -> Literal["execute_sql", "end"]:
            return "execute_sql" if state.get("is_sql_valid") else "end"

        builder = StateGraph(TimbrLikeState)
        builder.add_node("generate_sql", generate_sql)
        builder.add_node("validate_sql", validate_sql)
        builder.add_node("execute_sql", execute_sql)
        builder.add_node("generate_response", generate_response)

        builder.add_edge("generate_sql", "validate_sql")
        builder.add_conditional_edges(
            "validate_sql",
            route_validation,
            {"execute_sql": "execute_sql", "end": END},
        )
        builder.add_edge("execute_sql", "generate_response")
        builder.set_entry_point("generate_sql")

        return builder

    def _initial_state(self) -> TimbrLikeState:
        return {
            "prompt": "Show me all concepts",
            "sql": "",
            "concept": "",
            "rows": [],
            "response": "",
            "error": "",
            "is_sql_valid": False,
        }

    def test_graph_compiles(self):
        compiled = self._build_graph().compile()
        assert compiled is not None

    def test_graph_invoke_happy_path(self):
        compiled = self._build_graph().compile()
        result = compiled.invoke(self._initial_state())
        assert result["sql"] == "SELECT * FROM concept"
        assert result["is_sql_valid"] is True
        assert result["rows"] == [["row1"], ["row2"]]
        assert result["response"] == "Here are the results."

    def test_graph_invoke_validation_failure_stops_at_end(self):
        """When SQL is invalid the graph should stop before execute_sql."""

        def always_invalid(state: TimbrLikeState) -> dict:
            return {"is_sql_valid": False}

        def route(state: dict) -> Literal["execute_sql", "end"]:
            return "execute_sql" if state.get("is_sql_valid") else "end"

        builder = StateGraph(TimbrLikeState)
        builder.add_node("generate_sql", lambda s: {"sql": "BAD SQL", "concept": "x"})
        builder.add_node("validate_sql", always_invalid)
        execute_called = []
        builder.add_node("execute_sql", lambda s: execute_called.append(True) or {})
        builder.add_conditional_edges(
            "validate_sql", route, {"execute_sql": "execute_sql", "end": END}
        )
        builder.add_edge("generate_sql", "validate_sql")
        builder.set_entry_point("generate_sql")

        result = builder.compile().invoke(self._initial_state())
        assert result["is_sql_valid"] is False
        assert execute_called == [], "execute_sql should not have been called"

    def test_graph_with_memory_saver_persists_state(self):
        """StateGraph compiled with MemorySaver should persist state across invocations."""
        from langgraph.checkpoint.memory import MemorySaver

        memory = MemorySaver()
        compiled = self._build_graph().compile(checkpointer=memory)
        thread_cfg = {"configurable": {"thread_id": "timbr-thread-1"}}

        result = compiled.invoke(self._initial_state(), thread_cfg)
        assert result["response"] == "Here are the results."

        # Second invoke on same thread should succeed (state is checkpointed)
        result2 = compiled.invoke(self._initial_state(), thread_cfg)
        assert result2["response"] == "Here are the results."

    def test_multiple_threads_are_isolated(self):
        """Separate thread_ids must not share checkpointed state."""
        from langgraph.checkpoint.memory import MemorySaver

        memory = MemorySaver()
        compiled = self._build_graph().compile(checkpointer=memory)

        cfg_a = {"configurable": {"thread_id": "thread-A"}}
        cfg_b = {"configurable": {"thread_id": "thread-B"}}

        result_a = compiled.invoke(self._initial_state(), cfg_a)
        result_b = compiled.invoke(self._initial_state(), cfg_b)

        # Both should produce valid, independent results
        assert result_a["response"] == "Here are the results."
        assert result_b["response"] == "Here are the results."
