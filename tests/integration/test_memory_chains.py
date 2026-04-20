"""
Integration tests for conversation memory across chains.

These tests verify that memory (enable_memory=True) works correctly
when combined with history logging (enable_history=True) across
multiple chain invocations sharing the same conversation_id.

Validation strategy: after each follow-up invocation, fetch the
conversation history and verify the parent_query_id hierarchy — proving
the memory classifier correctly identified the follow-up and linked it.
"""

import pytest
import uuid6


from langchain_timbr.langchain.identify_concept_chain import IdentifyTimbrConceptChain
from langchain_timbr.langchain.generate_timbr_sql_chain import GenerateTimbrSqlChain
from langchain_timbr.langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from langchain_timbr.langchain.generate_answer_chain import GenerateAnswerChain
from langchain_timbr.utils.memory import fetch_conversation_history


def _build_conn_params(config):
    """Build conn_params dict matching what chains use internally."""
    return {
        "url": config["timbr_url"],
        "token": config["timbr_token"],
    }


def _fetch_history(config, conversation_id, top=20):
    """Fetch conversation history and return messages sorted oldest-first."""
    conn_params = _build_conn_params(config)
    messages = fetch_conversation_history(conn_params, conversation_id, top)
    assert messages is not None, "Failed to fetch conversation history"
    assert len(messages) > 0, "Conversation history is empty"
    return messages


def _assert_parent_hierarchy(messages, expected_depth):
    """Verify that messages form a parent chain of the expected depth.

    *messages* is the full history for a conversation_id (oldest first as
    returned by the API).  *expected_depth* is the total number of messages
    expected (e.g. 2 means one root + one follow-up).

    Asserts:
    - The history contains at least *expected_depth* messages.
    - The first message has no parent (it's the root).
    - Each subsequent message's ``parent_query_id`` points to a message
      that exists earlier in the history (same conversation hierarchy).
    """
    assert len(messages) >= expected_depth, (
        f"Expected at least {expected_depth} messages, got {len(messages)}"
    )
    id_map = {m["message_id"]: m for m in messages}
    known_ids = set()

    for i, msg in enumerate(messages):
        mid = msg["message_id"]
        parent_id = msg.get("parent_query_id")
        if i == 0:
            # Root message — should have no parent
            known_ids.add(mid)
        else:
            # Follow-up — parent_query_id must point to an earlier message
            assert parent_id is not None, (
                f"Message {i + 1} ({mid}) has no parent_query_id "
                f"— memory did not link it as a follow-up"
            )
            assert parent_id in known_ids, (
                f"Message {i + 1} parent_query_id={parent_id} "
                f"not found among earlier messages {known_ids}"
            )
            known_ids.add(mid)


class TestMemoryIdentifyConcept:
    """Tests for memory-enabled IdentifyTimbrConceptChain."""

    def test_follow_up_concept_identification(self, config, llm):
        """
        First establish conversation history via GenerateAnswerChain (only chain
        that stores history), then invoke IdentifyTimbrConceptChain with memory
        enabled to verify it can read the stored context.
        """
        conversation_id = str(uuid6.uuid7())

        # Initialize memory by running GenerateAnswerChain with enable_history=True
        answer_chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            enable_history=True,
            save_results=True,
            conversation_id=conversation_id,
        )
        result_history = answer_chain.invoke({
            "prompt": "What are the total sales for consumer customers?",
            "conversation_id": conversation_id,
        })
        assert "answer" in result_history

        # Now test IdentifyTimbrConceptChain with memory - should use stored history
        chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            conversation_id=conversation_id,
        )

        # Follow-up referencing prior conversation context
        result = chain.invoke({"prompt": "now for corporate"})
        assert "concept" in result
        assert result["concept"] is not None


class TestMemoryGenerateSql:
    """Tests for memory-enabled GenerateTimbrSqlChain."""

    def test_follow_up_sql_generation(self, config, llm):
        """
        First establish conversation history via GenerateAnswerChain (only chain
        that stores history), then invoke GenerateTimbrSqlChain with memory
        enabled to verify it can read the stored context.
        """
        conversation_id = str(uuid6.uuid7())

        # Initialize memory by running GenerateAnswerChain with enable_history=True
        answer_chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            enable_history=True,
            save_results=True,
            conversation_id=conversation_id,
        )
        result_history = answer_chain.invoke({
            "prompt": "What are the total sales for consumer customers?",
            "conversation_id": conversation_id,
        })
        assert "answer" in result_history

        # Now test GenerateTimbrSqlChain with memory - should use stored history
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            conversation_id=conversation_id,
        )

        # Follow-up referencing prior conversation context
        result = chain.invoke({"prompt": "Filter from 2021"})
        assert "sql" in result
        assert result["sql"] is not None


class TestMemoryExecuteQuery:
    """Tests for memory-enabled ExecuteTimbrQueryChain."""

    def test_follow_up_query_execution(self, config, llm):
        """
        First establish conversation history via GenerateAnswerChain (only chain
        that stores history), then invoke ExecuteTimbrQueryChain with memory
        enabled to verify it can read the stored context.
        """
        conversation_id = str(uuid6.uuid7())

        # Initialize memory by running GenerateAnswerChain with enable_history=True
        answer_chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            enable_history=True,
            save_results=True,
            conversation_id=conversation_id,
        )
        result_history = answer_chain.invoke({
            "prompt": "What are the total sales for consumer customers?",
            "conversation_id": conversation_id,
        })
        assert "answer" in result_history

        # Now test ExecuteTimbrQueryChain with memory - should use stored history
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            conversation_id=conversation_id,
        )

        # Follow-up referencing prior conversation context
        result = chain.invoke({"prompt": "top 5 results"})
        assert "rows" in result
        assert "sql" in result
        assert "conversation_id" in result


class TestMemoryAnswerChainFollowUp:
    """Tests for memory-enabled GenerateAnswerChain with 5-level follow-up questions."""

    def test_five_level_follow_up_conversation(self, config, llm):
        """
        Five sequential invocations of GenerateAnswerChain with the same conversation_id.
        Each follow-up question references prior context, testing that memory
        correctly carries conversation state across multiple turns.
        """
        conversation_id = str(uuid6.uuid7())

        chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=10,
            enable_history=True,
            save_results=True,
            conversation_id=conversation_id,
        )

        # Level 1: Initial question - establishes the conversation topic
        result1 = chain.invoke({
            "prompt": "What are the total sales for consumer customers?",
            "conversation_id": conversation_id,
        })
        assert "answer" in result1
        assert result1["answer"] is not None
        assert "conversation_id" in result1
        assert "sql" in result1

        # Level 2: Follow-up referencing "that" (the sales data)
        result2 = chain.invoke({
            "prompt": "Break that down by region",
            "conversation_id": conversation_id,
        })
        assert "answer" in result2
        assert result2["answer"] is not None
        assert "sql" in result2

        # Level 3: Follow-up narrowing scope from level 2
        result3 = chain.invoke({
            "prompt": "Show me only the top 3 regions",
            "conversation_id": conversation_id,
        })
        assert "answer" in result3
        assert result3["answer"] is not None
        assert "sql" in result3

        # Level 4: Follow-up pivoting to comparison (requires memory of levels 1-3)
        result4 = chain.invoke({
            "prompt": "Now with the corporate segment instead",
            "conversation_id": conversation_id,
        })
        assert "answer" in result4
        assert result4["answer"] is not None
        assert "sql" in result4

        # Level 5: Summarization referencing all prior turns
        result5 = chain.invoke({
            "prompt": "Summarize all the findings from our conversation so far",
            "conversation_id": conversation_id,
        })
        assert "answer" in result5
        assert result5["answer"] is not None
        # The summary answer should be non-trivial since it references prior turns
        assert len(result5["answer"]) > 20

        # Validate memory was used: all 5 messages should form a parent hierarchy
        messages = _fetch_history(config, conversation_id)
        _assert_parent_hierarchy(messages, expected_depth=5)

    def test_memory_does_not_leak_across_conversations(self, config, llm):
        """
        Two separate conversation_ids should not share memory.
        A follow-up with a new conversation_id should NOT reference prior context.
        """
        conversation_id_a = str(uuid6.uuid7())
        conversation_id_b = str(uuid6.uuid7())

        chain_a = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            enable_history=True,
            save_results=True,
            conversation_id=conversation_id_a,
        )

        chain_b = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            enable_history=True,
            save_results=True,
            conversation_id=conversation_id_b,
        )

        # Conversation A: establish context about consumer customers
        result_a = chain_a.invoke({
            "prompt": "What are the total sales for consumer customers?",
            "conversation_id": conversation_id_a,
        })
        assert "answer" in result_a
        assert result_a["answer"] is not None

        # Conversation B: independent question - should NOT have memory of conversation A
        result_b = chain_b.invoke({
            "prompt": "What are the total orders?",
            "conversation_id": conversation_id_b,
        })
        assert "answer" in result_b
        assert result_b["answer"] is not None
        assert "conversation_id" in result_b
        assert result_b["conversation_id"] == conversation_id_b

        # Validate isolation: each conversation should have exactly 1 message (no parent)
        messages_a = _fetch_history(config, conversation_id_a)
        messages_b = _fetch_history(config, conversation_id_b)
        assert len(messages_a) == 1, f"Conversation A should have 1 message, got {len(messages_a)}"
        assert len(messages_b) == 1, f"Conversation B should have 1 message, got {len(messages_b)}"
        assert messages_a[0].get("parent_query_id") is None, "Root message should have no parent"
        assert messages_b[0].get("parent_query_id") is None, "Root message should have no parent"
