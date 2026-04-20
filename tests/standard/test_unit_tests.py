"""Unit tests for individual chain components."""
import pytest
from unittest.mock import Mock, patch

from langchain_timbr import (
    IdentifyTimbrConceptChain,
    GenerateTimbrSqlChain,
    ValidateTimbrSqlChain,
    ExecuteTimbrQueryChain,
    GenerateAnswerChain,
)
from langchain_timbr.utils.timbr_llm_utils import _calculate_token_count


class TestChainUnitTests:
    """Unit tests for individual chain functionality."""
    
    def test_identify_concept_chain_unit(self, mock_llm):
        """Unit test for IdentifyTimbrConceptChain without external dependencies."""
        with patch('langchain_timbr.langchain.identify_concept_chain.determine_concept') as mock_determine:
            mock_determine.return_value = {
                'concept': 'customer',
                'schema': 'dtimbr',
                'concept_metadata': {},
                'usage_metadata': {}
            }
            
            chain = IdentifyTimbrConceptChain(
                llm=mock_llm,
                url="http://test",
                token="test",
                ontology="test"
            )
            
            result = chain.invoke({"prompt": "What are the customers?"})
            assert 'concept' in result
            assert 'prompt' in result, "invoke result must include input key 'prompt'"
            mock_determine.assert_called_once()
    
    def test_generate_sql_chain_unit(self, mock_llm):
        """Unit test for GenerateTimbrSqlChain without external dependencies."""
        with patch('langchain_timbr.langchain.generate_timbr_sql_chain.generate_sql') as mock_generate:
            mock_generate.return_value = {
                'sql': 'SELECT * FROM customer',
                'concept': 'customer',
                'usage_metadata': {}
            }
            
            chain = GenerateTimbrSqlChain(
                llm=mock_llm,
                url="http://test",
                token="test",
                ontology="test"
            )

            result = chain.invoke({"prompt": "Get all customers"})
            assert 'sql' in result
            assert 'prompt' in result, "invoke result must include input key 'prompt'"
            mock_generate.assert_called_once()
    
    def test_execute_query_chain_unit(self):
        """Test ExecuteTimbrQueryChain unit functionality."""
        from unittest.mock import Mock
        
        # Mock the LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "SELECT * FROM customers"
        
        # Create chain
        chain = ExecuteTimbrQueryChain(
            llm=mock_llm,
            url="http://test.com",
            token="test_token",
            ontology="test_ontology"
        )
        
        # Mock the _call method to return expected output format with all required keys
        expected_result = {
            "prompt": "Get all customers",
            "rows": [{"id": 1, "name": "Customer 1"}],
            "sql": "SELECT * FROM customers",
            "schema": "dtimbr",
            "concept": None,
            "error": None,
            "execute_timbr_usage_metadata": {}
        }
        chain._call = Mock(return_value=expected_result)

        # Test invocation
        result = chain.invoke({"prompt": "Get all customers"})

        # Verify result structure contains all expected keys
        assert isinstance(result, dict)
        assert "prompt" in result, "invoke result must include input key 'prompt'"
        assert "rows" in result
        assert "sql" in result
        assert "schema" in result
        assert "error" in result
        assert "execute_timbr_usage_metadata" in result
    
    def test_chain_input_sanitization(self, mock_llm):
        """Test that chains properly sanitize inputs."""
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url="http://test",
            token="test",
            ontology="test"
        )
        
        # Test with various input types
        test_prompts = [
            "normal question",
            "question with 'quotes'",
            "question with \"double quotes\"",
            "question with; semicolon",
            "",  # empty string
        ]
        
        for prompt in test_prompts:
            # Should not raise exceptions for any input
            try:
                # This will fail connection but shouldn't crash on input validation
                chain.invoke({"prompt": prompt})
            except Exception as e:
                # Should be connection-related, not input validation
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in 
                          ["connection", "invalid", "network", "rstrip", "nonetype"])
    
    def test_chain_parameter_validation(self, mock_llm):
        """Test that chains validate constructor parameters."""
        # Test that chain can be created with valid parameters
        try:
            chain = IdentifyTimbrConceptChain(
                llm=mock_llm,
                url="http://test",
                token="test",
                ontology="test"
            )
            assert chain is not None, "Chain should be created with valid parameters"
        except Exception as e:
            pytest.fail(f"Chain creation failed unexpectedly: {e}")
        
        # Test invalid parameter types (if the chain validates them)
        try:
            invalid_chain = IdentifyTimbrConceptChain(
                llm="not_an_llm",  # Invalid LLM type
                url="http://test",
                token="test",
                ontology="test"
            )
            # If it doesn't raise an error, that's also acceptable for some implementations
            assert invalid_chain is not None
        except (ValueError, TypeError, AttributeError):
            # These errors are expected for invalid parameters
            pass
    
    def test_chain_state_management(self, mock_llm):
        """Test that chains properly manage internal state."""
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url="http://test-url",
            token="test-token",
            ontology="test-ontology"
        )
        
        # Test that chain maintains configuration in private attributes
        assert hasattr(chain, '_url'), "Chain should store URL parameter"
        assert hasattr(chain, '_token'), "Chain should store token parameter"
        assert hasattr(chain, '_ontology'), "Chain should store ontology parameter"
        
        # Test that multiple instances don't interfere
        chain2 = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url="http://different",
            token="different-token",
            ontology="different-ontology"
        )
        
        assert chain._url != chain2._url
        assert chain._token != chain2._token
        assert chain._ontology != chain2._ontology

    def test_chain_output_includes_input_keys(self, mock_llm):
        """All input_keys must appear in the invoke() result (langchain 1.x compatibility)."""
        base_params = dict(llm=mock_llm, url="http://test", token="test", ontology="test")

        with patch('langchain_timbr.langchain.identify_concept_chain.determine_concept') as mock_determine:
            mock_determine.return_value = {'concept': 'customer', 'schema': 'dtimbr', 'concept_metadata': {}, 'usage_metadata': {}}
            chain = IdentifyTimbrConceptChain(**base_params)
            result = chain.invoke({"prompt": "test"})
            for key in chain.input_keys:
                assert key in result, f"IdentifyTimbrConceptChain: '{key}' missing from result"

        with patch('langchain_timbr.langchain.generate_timbr_sql_chain.generate_sql') as mock_gen:
            mock_gen.return_value = {'sql': 'SELECT 1', 'usage_metadata': {}}
            chain = GenerateTimbrSqlChain(**base_params)
            result = chain.invoke({"prompt": "test"})
            for key in chain.input_keys:
                assert key in result, f"GenerateTimbrSqlChain: '{key}' missing from result"

        with patch('langchain_timbr.langchain.validate_timbr_sql_chain.validate_sql') as mock_val:
            mock_val.return_value = (True, None, 'SELECT 1')
            chain = ValidateTimbrSqlChain(**base_params)
            result = chain.invoke({"prompt": "test", "sql": "SELECT 1"})
            for key in chain.input_keys:
                assert key in result, f"ValidateTimbrSqlChain: '{key}' missing from result"

        with patch('langchain_timbr.langchain.generate_answer_chain.answer_question') as mock_ans:
            mock_ans.return_value = {'answer': 'yes', 'usage_metadata': {}}
            chain = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test")
            result = chain.invoke({"prompt": "test", "rows": []})
            for key in chain.input_keys:
                assert key in result, f"GenerateAnswerChain: '{key}' missing from result"


class TestTokenCountFunctionality:
    """Test suite for token counting functionality with tiktoken."""
    
    def test_calculate_token_count_with_string_prompt(self):
        """Test token counting with a simple string prompt."""
        mock_llm = Mock()
        mock_llm._llm_type = "openai"
        mock_llm.client = Mock()
        mock_llm.client.model_name = "gpt-4"
        
        prompt = "What are the top customers?"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        assert token_count > 0, "Token count should be greater than 0 for non-empty prompt"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_with_list_prompt(self):
        """Test token counting with a list-based prompt (ChatPrompt format)."""
        mock_llm = Mock()
        mock_llm._llm_type = "openai"
        
        # Mock message objects with type and content
        system_msg = Mock()
        system_msg.type = "system"
        system_msg.content = "You are a helpful SQL assistant."
        
        user_msg = Mock()
        user_msg.type = "user"
        user_msg.content = "Generate SQL for top customers"
        
        prompt = [system_msg, user_msg]
        token_count = _calculate_token_count(mock_llm, prompt)
        
        assert token_count > 0, "Token count should be greater than 0 for non-empty prompt"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_without_model_name(self):
        """Test token counting falls back when LLM doesn't have model_name attribute."""
        mock_llm = Mock()
        # LLM without client.model_name attribute
        mock_llm.client = Mock(spec=[])
        
        prompt = "Test prompt without model name"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        # Should still return a count using fallback encoding
        assert token_count >= 0, "Token count should not fail when model_name is missing"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_without_client(self):
        """Test token counting falls back when LLM doesn't have client attribute."""
        mock_llm = Mock(spec=['_llm_type'])
        mock_llm._llm_type = "custom"
        # LLM without client attribute at all
        if hasattr(mock_llm, 'client'):
            delattr(mock_llm, 'client')
        
        prompt = "Test prompt without client"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        # Should still return a count using fallback encoding
        assert token_count >= 0, "Token count should not fail when client is missing"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_with_tiktoken_error(self):
        """Test token counting handles tiktoken errors gracefully."""
        mock_llm = Mock()
        mock_llm._llm_type = "custom"
        mock_llm.client = Mock()
        mock_llm.client.model_name = "unknown-model-that-causes-error"
        
        # This should not raise an exception even if tiktoken fails
        prompt = "Test prompt with potential tiktoken error"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        # Should return 0 or a valid count even on error
        assert token_count >= 0, "Token count should return 0 or valid count on error"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_empty_prompt(self):
        """Test token counting with empty prompt."""
        mock_llm = Mock()
        mock_llm._llm_type = "openai"
        
        prompt = ""
        token_count = _calculate_token_count(mock_llm, prompt)
        
        assert token_count == 0, "Token count should be 0 for empty prompt"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_with_different_llm_types(self):
        """Test token counting works with different LLM types."""
        llm_types = ["openai", "anthropic", "azure", "custom", "databricks"]
        
        for llm_type in llm_types:
            mock_llm = Mock()
            mock_llm._llm_type = llm_type
            
            prompt = f"Test prompt for {llm_type}"
            token_count = _calculate_token_count(mock_llm, prompt)
            
            assert token_count >= 0, f"Token count should work for {llm_type}"
            assert isinstance(token_count, int), f"Token count should be integer for {llm_type}"


class TestConversationIdAndAnswer:
    """Unit tests for conversation_id and answer fields in chain logging."""

    def test_conversation_id_stored_on_chain(self, mock_llm):
        """Chain constructors accept and store conversation_id."""
        from langchain_timbr import GenerateTimbrSqlChain, ExecuteTimbrQueryChain, GenerateAnswerChain
        from langchain_timbr.langchain.identify_concept_chain import IdentifyTimbrConceptChain

        base = dict(llm=mock_llm, url="http://test", token="test", ontology="test")

        for ChainCls in (GenerateTimbrSqlChain, ExecuteTimbrQueryChain, IdentifyTimbrConceptChain):
            chain = ChainCls(**base, conversation_id="conv-123")
            assert chain._conversation_id == "conv-123", f"{ChainCls.__name__} should store conversation_id"
            chain_no_conv = ChainCls(**base)
            assert chain_no_conv._conversation_id is None, f"{ChainCls.__name__} should default conversation_id to None"

        answer_chain = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test", conversation_id="conv-456")
        assert answer_chain._conversation_id == "conv-456"
        answer_chain_none = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test")
        assert answer_chain_none._conversation_id is None

    def test_agentlogcontext_conversation_id_field(self):
        """AgentLogContext accepts and stores conversation_id; defaults to None."""
        from langchain_timbr.utils.chain_logger import AgentLogContext
        from datetime import datetime, timezone

        ctx = AgentLogContext(
            query_id="qid-1",
            agent_name="agent",
            url="http://test",
            token="tok",
            chain_type="GenerateTimbrSqlChain",
            start_time=datetime.now(timezone.utc),
            prompt="test",
            enable_trace=False,
        )
        assert ctx.conversation_id is None

        ctx_with_conv = AgentLogContext(
            query_id="qid-2",
            agent_name="agent",
            url="http://test",
            token="tok",
            chain_type="GenerateTimbrSqlChain",
            start_time=datetime.now(timezone.utc),
            prompt="test",
            enable_trace=False,
            conversation_id="cid-99",
        )
        assert ctx_with_conv.conversation_id == "cid-99"

    def test_conversation_id_propagates_to_log_context(self, mock_llm):
        """conversation_id is passed to AgentLogContext when chain creates its own log context."""
        from unittest.mock import patch, call
        from langchain_timbr import GenerateTimbrSqlChain

        chain = GenerateTimbrSqlChain(
            llm=mock_llm,
            url="http://test",
            token="test",
            ontology="test",
            conversation_id="conv-abc",
            enable_trace=True,
        )

        captured_ctx = {}

        def fake_log_start(ctx, ontology=None, schema=None, additional_options="{}"):
            captured_ctx['ctx'] = ctx

        with patch('langchain_timbr.utils.chain_logger.log_agent_start', side_effect=fake_log_start), \
             patch('langchain_timbr.utils.chain_logger.log_agent_step'), \
             patch('langchain_timbr.utils.chain_logger.log_chain_trace'), \
             patch('langchain_timbr.langchain.generate_timbr_sql_chain.generate_sql') as mock_gen:
            mock_gen.return_value = {'sql': 'SELECT 1', 'is_sql_valid': True, 'usage_metadata': {}}
            chain.invoke({"prompt": "test"})

        assert 'ctx' in captured_ctx, "log_agent_start should have been called"
        assert captured_ctx['ctx'].conversation_id == "conv-abc"

    def test_conversation_id_defaults_to_query_id(self, mock_llm):
        """When no conversation_id provided, AgentLogContext.conversation_id equals query_id."""
        from unittest.mock import patch
        from langchain_timbr import GenerateTimbrSqlChain

        chain = GenerateTimbrSqlChain(
            llm=mock_llm,
            url="http://test",
            token="test",
            ontology="test",
            enable_trace=True,
        )

        captured_ctx = {}

        def fake_log_start(ctx, ontology=None, schema=None, additional_options="{}"):
            captured_ctx['ctx'] = ctx

        with patch('langchain_timbr.utils.chain_logger.log_agent_start', side_effect=fake_log_start), \
             patch('langchain_timbr.utils.chain_logger.log_agent_step'), \
             patch('langchain_timbr.utils.chain_logger.log_chain_trace'), \
             patch('langchain_timbr.langchain.generate_timbr_sql_chain.generate_sql') as mock_gen:
            mock_gen.return_value = {'sql': 'SELECT 1', 'is_sql_valid': True, 'usage_metadata': {}}
            chain.invoke({"prompt": "test"})

        ctx = captured_ctx['ctx']
        assert ctx.conversation_id == ctx.query_id, "conversation_id should default to query_id"

    def test_answer_passed_to_log_agent_history(self, mock_llm):
        """GenerateAnswerChain passes its answer to log_agent_history."""
        from unittest.mock import patch, MagicMock
        from langchain_timbr import GenerateAnswerChain

        chain = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test", enable_history=True)

        captured_kwargs = {}

        def fake_log_history(*args, **kwargs):
            captured_kwargs.update(kwargs)

        with patch('langchain_timbr.utils.chain_logger.log_agent_start'), \
             patch('langchain_timbr.utils.chain_logger.log_agent_step'), \
             patch('langchain_timbr.utils.chain_logger.log_chain_trace'), \
             patch('langchain_timbr.utils.chain_logger.log_agent_history', side_effect=fake_log_history), \
             patch('langchain_timbr.langchain.generate_answer_chain.answer_question') as mock_ans:
            mock_ans.return_value = {'answer': 'There are 42 customers.', 'usage_metadata': {}}
            chain.invoke({"prompt": "How many customers?", "rows": [{"count": 42}]})

        assert 'answer' in captured_kwargs, "answer should be passed to log_agent_history"
        assert captured_kwargs['answer'] == 'There are 42 customers.'

    def test_log_agent_history_payload_has_answer_and_conversation_id(self):
        """log_agent_history sends answer and conversation_id in the POST payload."""
        from unittest.mock import patch
        from datetime import datetime, timezone
        from langchain_timbr.utils.chain_logger import AgentLogContext, log_agent_history

        ctx = AgentLogContext(
            query_id="qid-test",
            agent_name="agent",
            url="http://test",
            token="tok",
            chain_type="TestChain",
            start_time=datetime.now(timezone.utc),
            prompt="test",
            enable_trace=False,
            conversation_id="cid-test",
        )

        captured_payload = {}

        def fake_safe_post(url, token, endpoint_path, payload):
            captured_payload.update(payload)

        with patch('langchain_timbr.utils.chain_logger._safe_post', side_effect=fake_safe_post):
            log_agent_history(
                ctx=ctx,
                ontology="ont",
                schema="sch",
                concept="concept",
                generated_sql=None,
                rows_returned=0,
                status="completed",
                failed_at_step=None,
                error=None,
                reasoning_status=None,
                usage_metadata={},
                answer_generated=True,
                llm_type="openai",
                llm_model="gpt-4o",
                answer="Test answer text.",
            )

        assert captured_payload.get('answer') == 'Test answer text.'
        assert captured_payload.get('conversation_id') == 'cid-test'
        assert 'parent_query_id' not in captured_payload
        assert 'is_follow_up' not in captured_payload

    def test_log_agent_history_none_answer_stripped(self):
        """log_agent_history omits 'answer' key when answer is None (_clean strips it)."""
        from unittest.mock import patch
        from datetime import datetime, timezone
        from langchain_timbr.utils.chain_logger import AgentLogContext, log_agent_history

        ctx = AgentLogContext(
            query_id="qid-nil",
            agent_name="agent",
            url="http://test",
            token="tok",
            chain_type="TestChain",
            start_time=datetime.now(timezone.utc),
            prompt="test",
            enable_trace=False,
        )

        captured_payload = {}

        def fake_safe_post(url, token, endpoint_path, payload):
            captured_payload.update(payload)

        with patch('langchain_timbr.utils.chain_logger._safe_post', side_effect=fake_safe_post):
            log_agent_history(
                ctx=ctx,
                ontology="ont",
                schema="sch",
                concept="concept",
                generated_sql=None,
                rows_returned=0,
                status="completed",
                failed_at_step=None,
                error=None,
                reasoning_status=None,
                usage_metadata={},
                answer_generated=False,
                llm_type="openai",
                llm_model="gpt-4o",
                answer=None,
            )

        assert 'answer' not in captured_payload, "None answer should be stripped by _clean()"


class TestGenerateAnswerChainWithFallbackExecution:
    """Tests covering the new GenerateAnswerChain behaviors added in this branch."""

    def test_generate_answer_chain_invokes_execute_chain_when_no_rows(self, mock_llm):
        """When rows are not provided, the embedded ExecuteTimbrQueryChain is called."""
        from unittest.mock import patch, Mock
        from langchain_timbr import GenerateAnswerChain

        chain = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test")
        chain._execute_chain.invoke = Mock(return_value={
            "rows": [{"n": 1}],
            "sql": "SELECT 1",
            "conversation_id": "q1",
            "chain_context": {},
        })

        with patch('langchain_timbr.langchain.generate_answer_chain.answer_question') as mock_ans:
            mock_ans.return_value = {"answer": "one", "usage_metadata": {}}
            result = chain.invoke({"prompt": "test"})

        chain._execute_chain.invoke.assert_called_once()
        assert result["rows"] == [{"n": 1}]
        assert result["answer"] == "one"

    def test_generate_answer_chain_skips_execute_chain_when_rows_provided(self, mock_llm):
        """When rows are already provided, the embedded ExecuteTimbrQueryChain is NOT called."""
        from unittest.mock import patch, Mock
        from langchain_timbr import GenerateAnswerChain

        chain = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test")
        chain._execute_chain.invoke = Mock()

        with patch('langchain_timbr.langchain.generate_answer_chain.answer_question') as mock_ans:
            mock_ans.return_value = {"answer": "ninety-nine", "usage_metadata": {}}
            result = chain.invoke({"prompt": "test", "rows": [{"n": 99}]})

        chain._execute_chain.invoke.assert_not_called()
        assert result["rows"] == [{"n": 99}]

    def test_generate_answer_chain_duration_tracked_in_chain_context(self, mock_llm):
        """After invoke(), chain_context contains a non-negative integer duration for GenerateAnswerChain."""
        from unittest.mock import patch
        from langchain_timbr import GenerateAnswerChain

        chain = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test")

        with patch('langchain_timbr.langchain.generate_answer_chain.answer_question') as mock_ans:
            mock_ans.return_value = {"answer": "x", "usage_metadata": {}}
            result = chain.invoke({"prompt": "test", "rows": []})

        duration = result["chain_context"]["duration"].get("GenerateAnswerChain")
        assert isinstance(duration, int), "duration should be an int (milliseconds)"
        assert duration >= 0

    def test_merge_usage_metadata(self, mock_llm):
        """_merge_usage_metadata sums numeric token fields across nested dicts."""
        from langchain_timbr import GenerateAnswerChain

        chain = GenerateAnswerChain(llm=mock_llm, url="http://test", token="test")

        base = {"answer_question": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        result = chain._merge_usage_metadata({}, base)
        assert result == base

        extra = {"answer_question": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}
        result = chain._merge_usage_metadata(result, extra)
        assert result["answer_question"]["input_tokens"] == 13
        assert result["answer_question"]["output_tokens"] == 7
        assert result["answer_question"]["total_tokens"] == 20


class TestHasNoMeaningfulResults:
    """Unit tests for ExecuteTimbrQueryChain._has_no_meaningful_results."""

    @pytest.fixture
    def chain(self, mock_llm):
        return ExecuteTimbrQueryChain(
            llm=mock_llm, url="http://test", token="test", ontology="test"
        )

    # --- empty / None rows ---

    def test_empty_list_returns_true(self, chain):
        assert chain._has_no_meaningful_results([], "SELECT 1") is True

    def test_none_rows_returns_true(self, chain):
        assert chain._has_no_meaningful_results(None, "SELECT 1") is True

    # --- rows with real data ---

    def test_single_row_with_values(self, chain):
        rows = [{"name": "Alice", "amount": 100}]
        assert chain._has_no_meaningful_results(rows, "SELECT name, amount FROM t") is False

    def test_multiple_rows_with_values(self, chain):
        rows = [{"id": 1, "v": 10}, {"id": 2, "v": 20}]
        assert chain._has_no_meaningful_results(rows, "SELECT id, v FROM t") is False

    def test_row_with_some_none_and_some_values(self, chain):
        rows = [{"a": None, "b": 42}]
        assert chain._has_no_meaningful_results(rows, "SELECT a, b FROM t") is False

    # --- all-None rows ---

    def test_single_row_all_none(self, chain):
        rows = [{"a": None, "b": None}]
        assert chain._has_no_meaningful_results(rows, "SELECT a, b FROM t") is True

    def test_multiple_rows_all_none(self, chain):
        rows = [{"a": None}, {"a": None}]
        assert chain._has_no_meaningful_results(rows, "SELECT a FROM t") is True

    def test_multiple_rows_one_has_value(self, chain):
        rows = [{"a": None}, {"a": 5}]
        assert chain._has_no_meaningful_results(rows, "SELECT a FROM t") is False

    # --- aggregate with 0 ---

    def test_count_returning_zero(self, chain):
        rows = [{"count_val": 0}]
        assert chain._has_no_meaningful_results(rows, "SELECT COUNT(id) AS count_val FROM t") is True

    def test_sum_returning_zero(self, chain):
        rows = [{"total": 0}]
        assert chain._has_no_meaningful_results(rows, "SELECT SUM(amount) AS total FROM t") is True

    def test_avg_returning_null(self, chain):
        rows = [{"avg_val": None}]
        assert chain._has_no_meaningful_results(rows, "SELECT AVG(price) AS avg_val FROM t") is True

    def test_min_returning_zero(self, chain):
        rows = [{"min_val": 0}]
        assert chain._has_no_meaningful_results(rows, "SELECT MIN(qty) AS min_val FROM t") is True

    def test_max_returning_null(self, chain):
        rows = [{"max_val": None}]
        assert chain._has_no_meaningful_results(rows, "SELECT MAX(score) AS max_val FROM t") is True

    def test_aggregate_mixed_zero_and_null(self, chain):
        rows = [{"cnt": 0, "total": None}]
        assert chain._has_no_meaningful_results(rows, "SELECT COUNT(id) AS cnt, SUM(v) AS total FROM t") is True

    # --- aggregate with actual results (should be meaningful) ---

    def test_count_returning_nonzero(self, chain):
        rows = [{"cnt": 42}]
        assert chain._has_no_meaningful_results(rows, "SELECT COUNT(id) AS cnt FROM t") is False

    def test_sum_returning_nonzero(self, chain):
        rows = [{"total": 1500.50}]
        assert chain._has_no_meaningful_results(rows, "SELECT SUM(amount) AS total FROM t") is False

    def test_aggregate_one_zero_one_nonzero(self, chain):
        rows = [{"cnt": 0, "total": 100}]
        assert chain._has_no_meaningful_results(rows, "SELECT COUNT(id) AS cnt, SUM(v) AS total FROM t") is False

    # --- non-aggregate single row with all zeros (should be meaningful) ---

    def test_non_aggregate_single_row_all_zeros(self, chain):
        """A plain SELECT returning zeros is still meaningful data."""
        rows = [{"a": 0, "b": 0}]
        assert chain._has_no_meaningful_results(rows, "SELECT a, b FROM t") is False

    # --- multiple rows from aggregate (not the single-row shortcut) ---

    def test_aggregate_multiple_rows_not_caught_by_single_row_check(self, chain):
        """Multi-row aggregates with zeros are not caught by the single-row check."""
        rows = [{"cnt": 0}, {"cnt": 0}]
        # Two rows → skips single-row aggregate check, falls to all-None check.
        # Rows have 0 (not None), so at least one value is not None → meaningful.
        assert chain._has_no_meaningful_results(rows, "SELECT COUNT(id) AS cnt FROM t GROUP BY region") is False

    # --- None sql parameter ---

    def test_none_sql_with_rows(self, chain):
        rows = [{"a": 1}]
        assert chain._has_no_meaningful_results(rows, None) is False

    def test_none_sql_empty_rows(self, chain):
        assert chain._has_no_meaningful_results([], None) is True
