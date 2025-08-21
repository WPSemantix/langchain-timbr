Make sure to set this environment variables berore running the tests:
1. LLM_API_KEY
2. TIMBR_URL - the default is https://demo-env.timbr.ai
3. TIMBR_TOKEN
4. TIMBR_ONTOLOGY - the default is "supply_metrics_llm_tests"
5. TEST_PROMPT - the default is "What are the total sales for consumer customers?"

Execute tests using:
python -m pytest

Execute a specific test file:
python -m pytest <file_name>