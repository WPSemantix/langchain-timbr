from langchain_timbr import ExecuteTimbrQueryChain, GenerateAnswerChain


class TestChainPipeline:
    """Test suite for LangChain sequential chain pipeline functionality."""
    
    def test_chain_pipeline(self, llm, config):
        """Test sequential chain pipeline combining query execution and answer generation."""
        execute_timbr_query_chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )

        generate_answer_chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            verify_ssl=config["verify_ssl"],
        )

        result = execute_timbr_query_chain.invoke({ "prompt": config["test_prompt"] })
        answer_result = generate_answer_chain.invoke({
            "prompt": config["test_prompt"],
            "rows": result.get("rows"),
            "sql": result.get("sql"),
        })
        result.update(answer_result)
        print("Pipeline result:", result)
        assert "sql" in result, "Pipeline should return an 'sql'"
        assert "answer" in result, "Pipeline should return an 'answer'"
        assert result["answer"], "Answer should not be empty"
        assert execute_timbr_query_chain.usage_metadata_key in result, "Pipeline should return execute chain usage metadata"
        assert generate_answer_chain.usage_metadata_key in result, "Pipeline should return generate chain usage metadata"
  