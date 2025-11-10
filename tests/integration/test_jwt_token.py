import requests
from langchain_timbr import create_timbr_sql_agent


class TestTimbrSqlAgentJWTAuthentication:
    """Test suite for Timbr SQL Agent JWT authentication functionality."""
    
    # Due to policy changes and MFA requirements, this test is currently skipped (still running at timbr-chainlit repository).
    def skip_test_timbr_sql_agent_integration(self, llm, config):
        """Test Timbr SQL Agent integration with JWT authentication."""
        # Azure AD token endpoint URL
        token_url = f'https://login.microsoftonline.com/{config["jwt_tenant_id"]}/oauth2/v2.0/token'

        # Request payload for token exchange
        payload = {
            'client_id': config["jwt_client_id"],
            'client_secret': config["jwt_secret"],
            'scope': config["jwt_scope"],
            'username': config["jwt_username"],
            'password': config["jwt_password"],
            'grant_type': 'password'
        }

        # Request headers
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        # Make the request to get the access token
        response = requests.post(token_url, data=payload, headers=headers)
        tokens = response.json()

        access_token = None
        if response.status_code == 200:
            access_token = tokens.get('access_token')
            print(f"Access Token: {access_token}")
        else:
            print(f"Error fetching access token: {tokens}")
            assert False, f"Error fetching access token: {tokens}"

        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["jwt_timbr_url"],
            token=access_token,
            ontology=config["jwt_timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            is_jwt=True,
        )
        result = agent.invoke("show one product")
        
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "rows" in result, "Rows should be returned"
        assert "concept" in result, "Concept should be returned"
        assert len(result["rows"]) > 0, "Rows should be returned"
        assert "usage_metadata" in result, "Agent should return 'usage_metadata'"
        assert len(result["usage_metadata"]) == 2 and 'determine_concept' in result["usage_metadata"] and 'generate_sql' in result["usage_metadata"], "Usage metadata should contain both 'determine_concept' and 'generate_sql'"


