import pytest

from langchain_timbr import (
    IdentifyTimbrConceptChain,
    GenerateTimbrSqlChain,
    ValidateTimbrSqlChain,
    ExecuteTimbrQueryChain,
    GenerateAnswerChain
)
from langchain_timbr.utils import timbr_llm_utils


class TestIdentifyTimbrConceptChain:
    """Test suite for IdentifyTimbrConceptChain functionality."""
    
    def test_identify_concept_chain(self, llm, config):
        """Test basic concept identification functionality."""
        chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("IdentifyTimbrConceptChain result:", result)
        assert "concept" in result, "Chain should return a 'concept'"
        assert result["concept"], "Returned concept should not be empty"
        assert chain.usage_metadata_key in result, "Chain should return 'usage_metadata'"
        assert len(result[chain.usage_metadata_key]) == 1 and 'determine_concept' in result[chain.usage_metadata_key], "Usage metadata should contain only 'determine_concept'"

    def test_identify_concept_chain_none_lists(self, llm, config):
        """Test concept identification with empty concept/view lists raises exception."""
        chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            concepts_list='none',
            views_list='null',
            verify_ssl=config["verify_ssl"],
        )
        with pytest.raises(Exception, match="No relevant concepts found for the query"):
            result = chain.invoke({ "prompt": config["test_prompt"] })


class TestGenerateTimbrSqlChain:
    """Test suite for GenerateTimbrSqlChain functionality."""
    
    def test_generate_timbr_sql_chain(self, llm, config):
        """Test basic SQL generation functionality."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("GenerateTimbrSqlChain result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"], "Concept name should be returned"
        assert chain.usage_metadata_key in result, "Chain should return 'usage_metadata'"
        assert len(result[chain.usage_metadata_key]) == 2 and 'determine_concept' in result[chain.usage_metadata_key] and 'generate_sql' in result[chain.usage_metadata_key], "Usage metadata should contain both 'determine_concept' and 'generate_sql'"

    def test_generate_timbr_sql_with_limit_chain(self, llm, config):
        """Test SQL generation with row limit."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            max_limit=2,
            verify_ssl=config["verify_ssl"],
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("GenerateTimbrSqlChain with max_limit=2 result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "LIMIT 2" in result["sql"].upper(), "SQL should contain 'LIMIT 2'"

    def test_generate_timbr_sql_with_concepts_list_chain(self, llm, config):
        """Test SQL generation with specific concepts list."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            concepts_list=['plant'],
            verify_ssl=config["verify_ssl"],
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("GenerateTimbrSqlChain with concepts_list=['plant'] result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "`dtimbr`.`plant`" in result["sql"], "SQL should reference '`dtimbr`.`plant`'"
        assert "concept" in result and result["concept"], "Concept name should be returned"
        assert result["concept"] == "plant", "Concept should equal 'plant'"

    def test_generate_timbr_sql_excluding_builtin_columns(self, llm, config):
        """Test SQL generation excluding built-in columns."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            max_limit=1,
            verify_ssl=config["verify_ssl"],
        )
        result = chain.invoke({ "prompt": config["test_prompt_2"] })
        print("GenerateTimbrSqlChain excluding built-in columns result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        builtin_columns = ["entity_type", "entity_id", "entity_label"]
        for col in builtin_columns:
            assert col not in result["sql"].lower(), f"SQL should not contain '{col}'"
        assert "concept" in result and result["concept"], "Concept name should be returned"

    def test_generate_timbr_sql_excluding_selected_columns(self, llm, config):
        """Test SQL generation excluding custom selected columns."""
        exclude_properties = ['customer_name', 'customer_segment']
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            exclude_properties=exclude_properties,
            max_limit=1,
            verify_ssl=config["verify_ssl"],
        )
        result = chain.invoke({ "prompt": config["test_prompt_2"] })
        print("GenerateTimbrSqlChain excluding selected columns result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
            
        for col in exclude_properties:
            assert col not in result["sql"].lower(), f"SQL should not contain '{col}'"
        assert "concept" in result and result["concept"], "Concept name should be returned"

    def test_generate_timbr_sql_chain_descriptions(self, llm, config):
        """Test SQL generation with concept descriptions in debug mode."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            debug=True
        )
        result = chain.invoke({ "prompt": config["test_prompt_2"] })
        print("GenerateTimbrSqlChain result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"] == "customer", "Concept customer should be returned"
        prompt = timbr_llm_utils.decrypt_prompt(result["generate_sql_usage_metadata"]["generate_sql"]["p_hash"], timbr_llm_utils.generate_key())
        assert "customer related info" in prompt, "Customer description should be in prompt"
        assert "concat of first and last name" in prompt, "Customer name description should be in prompt"
        assert "continent name" in prompt, "Order market description should be in prompt"
        assert "shipment of customer" in prompt, "Shipment relationship description should be in prompt"
        assert "count first names of customers" in prompt, "Customer first name count measure description should be in prompt"
        assert "calculation of revenue based on sales" in prompt, "Revenue measure description should be in prompt"

    def test_generate_timbr_sql_chain_descriptions_specific_concept(self, llm, config):
        """Test SQL generation with specific concept and descriptions."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            concept="customer",
            debug=True
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("GenerateTimbrSqlChain result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"] == "customer", "Concept customer should be returned"
        prompt = timbr_llm_utils.decrypt_prompt(result["generate_sql_usage_metadata"]["generate_sql"]["p_hash"], timbr_llm_utils.generate_key())
        assert "customer related info" in prompt, "Customer description should be in prompt"
        assert "concat of first and last name" in prompt, "Customer name description should be in prompt"
        assert "continent name" in prompt, "Order market description should be in prompt"
        assert "shipment of customer" in prompt, "Shipment relationship description should be in prompt"
        assert "count first names of customers" in prompt, "Customer first name count measure description should be in prompt"
        assert "calculation of revenue based on sales" in prompt, "Revenue measure description should be in prompt"

    def test_generate_timbr_sql_chain_descriptions_specific_cube(self, llm, config):
        """Test SQL generation with specific cube/view and descriptions."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            views_list=["customer_cube"],
            debug=True
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("GenerateTimbrSqlChain result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"] == "customer_cube", "Concept customer_cube should be returned"
        prompt = timbr_llm_utils.decrypt_prompt(result["generate_sql_usage_metadata"]["generate_sql"]["p_hash"], timbr_llm_utils.generate_key())
        assert "customer cube related info" in prompt, "Customer description should be in prompt"
        assert "concat of first and last name" in prompt, "Customer name description should be in prompt"
        assert "continent name" in prompt, "Order market description should be in prompt"
        assert "count first names of customers" in prompt, "Customer first name count measure description should be in prompt"
        assert "calculation of revenue based on sales" in prompt, "Revenue measure description should be in prompt"

    def test_generate_timbr_sql_chain_tags(self, llm, config):
        """Test SQL generation with tags filtering."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            include_tags=["length", "synonym"],
            debug=True
        )
        result = chain.invoke({ "prompt": config["test_prompt_3"] })
        print("GenerateTimbrSqlChain result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"] == "product", "Concept product should be returned"
        assert chain.usage_metadata_key in result, "Chain should return 'usage_metadata'"
        prompt = timbr_llm_utils.decrypt_prompt(result["generate_sql_usage_metadata"]["generate_sql"]["p_hash"], timbr_llm_utils.generate_key())
        assert "commodity" in prompt, "Product tag value of synonym should be in prompt"
        assert "synonym" in prompt, "Product tag synonym should be in prompt"
        assert "length" in prompt, "Material tag should be in prompt"
        assert "max length 10 characters" in prompt, "Material tag value should be in prompt"
        assert "count of commodities" in prompt, "Product count measure tag value should be in prompt"

    def test_generate_timbr_sql_chain_tags_specific_concept(self, llm, config):
        """Test SQL generation with tags filtering and specific concept."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            concept="product",
            include_tags=["length", "synonym"],
            debug=True
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("GenerateTimbrSqlChain result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"] == "product", "Concept product should be returned"
        assert chain.usage_metadata_key in result, "Chain should return 'usage_metadata'"
        prompt = timbr_llm_utils.decrypt_prompt(result["generate_sql_usage_metadata"]["generate_sql"]["p_hash"], timbr_llm_utils.generate_key())
        assert "commodity" in prompt, "Product tag value of synonym should be in prompt"
        assert "synonym" in prompt, "Product tag synonym should be in prompt"
        assert "length" in prompt, "Material tag should be in prompt"
        assert "max length 10 characters" in prompt, "Material tag value should be in prompt"
        assert "count of commodities" in prompt, "Product count measure tag value should be in prompt"
        assert "total product price" in prompt, "Product total price measure tag value should be in prompt"

    def test_generate_timbr_sql_chain_tags_specific_cube(self, llm, config):
        """Test SQL generation with tags filtering and specific cube/view."""
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            views_list=["product_cube"],
            include_tags=["length", "synonym"],
            debug=True
        )
        result = chain.invoke({ "prompt": config["test_prompt"] })
        print("GenerateTimbrSqlChain result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"] == "product_cube", "Concept product_cube should be returned"
        assert chain.usage_metadata_key in result, "Chain should return 'usage_metadata'"
        prompt = timbr_llm_utils.decrypt_prompt(result["generate_sql_usage_metadata"]["generate_sql"]["p_hash"], timbr_llm_utils.generate_key())
        assert "commodity cube" in prompt, "Product tag value of synonym should be in prompt"
        assert "synonym" in prompt, "Product tag synonym should be in prompt"
        assert "length" in prompt, "Material tag should be in prompt"
        assert "max length 10 characters" in prompt, "Material tag value should be in prompt"
        assert "count of commodities" in prompt, "Product count measure tag value should be in prompt"
        assert "total product price" in prompt, "Product total price measure tag value should be in prompt"


class TestValidateTimbrSqlChain:
    """Test suite for ValidateTimbrSqlChain functionality."""
    
    def test_validate_timbr_sql_chain(self, llm, config):
        """Test SQL validation functionality."""
        chain = ValidateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            retries=1,  # Use a single retry for test speed
            verify_ssl=config["verify_ssl"],
        )
        inputs = {
            "prompt": config["test_prompt"],
            "sql": "SELECT * FROM invalid_table",  # Intentionally invalid (or test with a valid one if available)
        }
        result = chain.invoke(inputs)
        print("ValidateTimbrSqlChain result:", result)
        # Check that we have a boolean flag and an error message (or None)
        assert "is_sql_valid" in result
        assert isinstance(result["is_sql_valid"], bool)
        # Optionally, assert that for a valid SQL the flag is True
        if result["sql"] and "invalid_table" not in result["sql"]:
            assert result["is_sql_valid"] is True


class TestExecuteTimbrQueryChain:
    """Test suite for ExecuteTimbrQueryChain functionality."""
    
    def test_execute_timbr_query_chain(self, llm, config):
        """Test basic query execution functionality."""
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )
        # For testing, use a prompt and SQL that are expected to produce results.
        inputs = {
            "prompt": config["test_prompt"],
        }
        result = chain.invoke(inputs)
        print("ExecuteTimbrQueryChain result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present in the result"

    def test_execute_timbr_query_chain_with_limit(self, llm, config):
        """Test query execution with row limit and tags."""
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            max_limit=3,
            include_tags="*",
            verify_ssl=config["verify_ssl"],
        )
        inputs = {
            "prompt": config["test_prompt"],
        }
        result = chain.invoke(inputs)
        print("ExecuteTimbrQueryChain with max_limit=3 result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present in the result"
        assert "LIMIT 3" in result["sql"].upper(), "SQL should contain 'LIMIT 3'"
        assert len(result["rows"]) <= 3, "Number of rows should not exceed max_limit"

    def test_execute_timbr_query_chain_with_result_inference(self, llm, config):
        """Test query execution with result inference and retry logic."""
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            retry_if_no_results=True,
            verify_ssl=config["verify_ssl"],
        )
        inputs = {
            "prompt": config["test_prompt"],
            "concept": "order",
            "schema": "dtimbr",
            "sql": "SELECT COUNT(1) AS `total_sales` FROM `dtimbr`.`order` WHERE order_city = 'Rishon LeZion'",
        }
        result = chain.invoke(inputs)
        print("ExecuteTimbrQueryChain with result inference result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present in the result"
        assert "total_sales" in result["rows"][0], "Result rows should contain 'total_sales'"


class TestGenerateAnswerChain:
    """Test suite for GenerateAnswerChain functionality."""
    
    def test_generate_answer_chain(self, llm, config):
        """Test basic answer generation functionality."""
        chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
        )
        result = chain.invoke({ "prompt": config["test_prompt"], "rows": [{ "total_sales": 1000 }] })
        print("GenerateAnswerChain result:", result)
        assert "answer" in result, "Chain should return an 'answer'"
        assert result["answer"], "Answer should not be empty"

    def test_generate_answer_chain_with_sql(self, llm, config):
        """Test answer generation with SQL context."""
        chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
        )
        inputs = {
            "prompt": config["test_prompt"],
            "rows": [{ "counter": 1000 }],
            "sql": "SELECT COUNT(1) AS `counter` FROM `dtimbr`.`order` WHERE order_city = 'Rishon LeZion'",
        }
        result = chain.invoke(inputs)
        print("GenerateAnswerChain with SQL result:", result)
        assert "answer" in result, "Chain should return an 'answer'"
        assert result["answer"], "Answer should not be empty"
        assert chain.usage_metadata_key in result, "Chain should return 'generate_answer_usage_metadata'"
        assert len(result[chain.usage_metadata_key]) == 1 and 'answer_question' in result[chain.usage_metadata_key], "Generate answer chain usage metadata should contain only 'answer_question'"