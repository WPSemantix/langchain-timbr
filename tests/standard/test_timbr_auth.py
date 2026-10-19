"""Unit tests for exchanging a JWT for a Timbr token (utils/timbr_auth.py). No real OAuth or Timbr calls."""
import pytest
from unittest.mock import Mock, patch

from langchain_timbr import config
from langchain_timbr.langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from langchain_timbr.timbr_llm_connector import TimbrLlmConnector
from langchain_timbr.utils import timbr_auth, timbr_utils
from langchain_timbr.utils.timbr_auth import exchange_jwt, resolve_timbr_auth

URL = "https://timbr.example"
JWT = "eyJ.header.payload-signature"
TIMBR_TOKEN = "tk_exchanged"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    exchange_jwt.cache_clear()
    timbr_auth._spn_timbr_token.cache_clear()
    for name in ("jwt_client_id", "jwt_client_secret", "jwt_scope", "jwt_tenant_id"):
        monkeypatch.setattr(config, name, None)
    yield
    exchange_jwt.cache_clear()
    timbr_auth._spn_timbr_token.cache_clear()


@pytest.fixture
def show_token():
    with patch.object(timbr_auth, "run_query", return_value=[{"token": TIMBR_TOKEN}]) as mock:
        yield mock


def _aad_response(status=200, body=None):
    response = Mock(status_code=status)
    response.json.return_value = body if body is not None else {"access_token": JWT}
    return response


class TestExchange:
    def test_timbr_token_is_not_exchanged(self, show_token):
        assert resolve_timbr_auth(URL, "tk_static", False) == ("tk_static", False)
        show_token.assert_not_called()

    def test_jwt_is_exchanged_for_timbr_token(self, show_token):
        assert resolve_timbr_auth(URL, JWT, True, "tenant-1") == (TIMBR_TOKEN, False)

        query, conn_params = show_token.call_args.args
        assert query == "SHOW TOKEN"
        assert conn_params["token"] == JWT
        assert conn_params["is_jwt"] is True
        assert conn_params["jwt_tenant_id"] == "tenant-1"

    def test_exchange_is_cached_per_jwt(self, show_token):
        resolve_timbr_auth(URL, JWT, True)
        resolve_timbr_auth(URL, JWT, True)
        assert show_token.call_count == 1

        resolve_timbr_auth(URL, JWT + "-other", True)
        assert show_token.call_count == 2

    def test_rejected_jwt_raises_clear_error(self):
        server_error = Exception("Error: Error running query: 401 Unauthorized: Error: Audience doesn't match")
        with patch.object(timbr_auth, "run_query", side_effect=server_error):
            with pytest.raises(ValueError) as exc_info:
                resolve_timbr_auth(URL, JWT, True)

        message = str(exc_info.value)
        assert "Timbr rejected the JWT token" in message
        assert "Audience doesn't match" in message
        assert JWT not in message

    def test_other_exchange_failure_raises(self):
        with patch.object(timbr_auth, "run_query", side_effect=Exception("Connection refused")):
            with pytest.raises(ValueError, match="Failed to exchange the JWT token.*Connection refused"):
                resolve_timbr_auth(URL, JWT, True)

    def test_failed_exchange_is_not_cached(self, show_token):
        show_token.side_effect = [Exception("timeout"), [{"token": TIMBR_TOKEN}]]
        with pytest.raises(ValueError):
            resolve_timbr_auth(URL, JWT, True)
        assert resolve_timbr_auth(URL, JWT, True) == (TIMBR_TOKEN, False)

    def test_empty_show_token_raises(self):
        with patch.object(timbr_auth, "run_query", return_value=[]):
            with pytest.raises(ValueError, match="SHOW TOKEN returned no token"):
                resolve_timbr_auth(URL, JWT, True)

    def test_no_url_returns_inputs_for_validation(self, show_token):
        assert resolve_timbr_auth(None, JWT, True) == (JWT, True)
        show_token.assert_not_called()


class TestServicePrincipalFromEnv:
    @pytest.fixture
    def spn_env(self, monkeypatch):
        monkeypatch.setattr(config, "jwt_tenant_id", "tenant-1")
        monkeypatch.setattr(config, "jwt_client_id", "client-1")
        monkeypatch.setattr(config, "jwt_client_secret", "s3cret")
        monkeypatch.setattr(config, "jwt_scope", "api://timbr-app/.default")

    def test_no_token_and_no_spn_env_is_left_to_validation(self, show_token):
        assert resolve_timbr_auth(URL, None, False) == (None, False)
        show_token.assert_not_called()

    def test_spn_env_gets_jwt_and_exchanges_it(self, spn_env, show_token):
        with patch.object(timbr_auth.requests, "post", return_value=_aad_response()) as post:
            assert resolve_timbr_auth(URL, None, False) == (TIMBR_TOKEN, False)

        assert post.call_args.args[0] == "https://login.microsoftonline.com/tenant-1/oauth2/v2.0/token"
        assert post.call_args.kwargs["data"] == {
            "client_id": "client-1",
            "client_secret": "s3cret",
            "scope": "api://timbr-app/.default",
            "grant_type": "client_credentials",
        }
        assert show_token.call_args.args[1]["token"] == JWT

    def test_spn_token_is_cached(self, spn_env, show_token):
        with patch.object(timbr_auth.requests, "post", return_value=_aad_response()) as post:
            resolve_timbr_auth(URL, None, False)
            resolve_timbr_auth(URL, None, False)
        assert post.call_count == 1
        assert show_token.call_count == 1

    def test_explicit_token_wins_over_spn_env(self, spn_env, show_token):
        with patch.object(timbr_auth.requests, "post") as post:
            assert resolve_timbr_auth(URL, "tk_static", False) == ("tk_static", False)
        post.assert_not_called()

    def test_partial_spn_env_lists_missing_vars(self, monkeypatch):
        monkeypatch.setattr(config, "jwt_client_secret", "s3cret")
        with pytest.raises(ValueError) as exc_info:
            resolve_timbr_auth(URL, None, False)
        message = str(exc_info.value)
        assert "JWT_TENANT_ID" in message and "JWT_CLIENT_ID" in message and "JWT_SCOPE" in message
        assert "JWT_CLIENT_SECRET" not in message
        assert "s3cret" not in message

    def test_spn_vars_without_secret_are_ignored(self, monkeypatch, show_token):
        # e.g. CI sets JWT_TENANT_ID / JWT_CLIENT_ID / JWT_SCOPE for another JWT flow
        monkeypatch.setattr(config, "jwt_tenant_id", "tenant-1")
        monkeypatch.setattr(config, "jwt_client_id", "client-1")
        monkeypatch.setattr(config, "jwt_scope", "api://timbr-app/.default")
        assert resolve_timbr_auth(URL, None, False) == (None, False)
        show_token.assert_not_called()

    def test_aad_failure_raises_without_secret(self, spn_env):
        body = {"error": "invalid_client", "error_description": "AADSTS7000215: Invalid client secret provided."}
        with patch.object(timbr_auth.requests, "post", return_value=_aad_response(401, body)):
            with pytest.raises(ValueError) as exc_info:
                resolve_timbr_auth(URL, None, False)
        message = str(exc_info.value)
        assert "invalid_client" in message and "HTTP 401" in message
        assert "s3cret" not in message


class TestChainsUseExchangedToken:
    def test_chain_sends_timbr_token_as_api_key(self, show_token):
        chain = ExecuteTimbrQueryChain(llm=Mock(), url=URL, token=JWT, is_jwt=True, jwt_tenant_id="tenant-1", ontology="ont")
        conn_params = chain._get_conn_params()
        assert conn_params["token"] == TIMBR_TOKEN
        assert conn_params["is_jwt"] is False

        session = Mock()
        with patch.object(timbr_utils, "_get_session", return_value=session), \
             patch.object(timbr_utils.timbr_http_connector, "_parse_response", return_value=[]):
            timbr_utils._run_query_pooled(query="SELECT 1", **{k: v for k, v in conn_params.items() if k != "additional_headers"})
        headers = session.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == TIMBR_TOKEN
        assert "x-jwt-token" not in headers

    def test_chain_with_timbr_token_is_unchanged(self, show_token):
        chain = ExecuteTimbrQueryChain(llm=Mock(), url=URL, token="tk_static", ontology="ont")
        assert chain._get_conn_params()["token"] == "tk_static"
        show_token.assert_not_called()

    def test_chain_without_token_still_raises_validation_error(self, monkeypatch, show_token):
        monkeypatch.setattr(config, "token", None)
        with pytest.raises(ValueError, match="Token must be provided"):
            ExecuteTimbrQueryChain(llm=Mock(), url=URL, ontology="ont")

    def test_jwt_inside_conn_params_is_exchanged(self, show_token):
        chain = ExecuteTimbrQueryChain(
            llm=Mock(), url=URL, token="tk_static", ontology="ont",
            conn_params={"token": JWT, "is_jwt": True},
        )
        conn_params = chain._get_conn_params()
        assert conn_params["token"] == TIMBR_TOKEN
        assert conn_params["is_jwt"] is False

    def test_llm_connector_exchanges_jwt(self, show_token):
        connector = TimbrLlmConnector(llm=Mock(), url=URL, token=JWT, is_jwt=True)
        conn_params = connector._get_conn_params()
        assert conn_params["token"] == TIMBR_TOKEN
        assert conn_params["is_jwt"] is False
