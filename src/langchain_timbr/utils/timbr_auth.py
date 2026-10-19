"""Exchange an Azure AD / Entra ID JWT for the caller's Timbr token.

A JWT is exchanged once, with ``SHOW TOKEN``, for the Timbr token of the user it
belongs to. Timbr tokens do not expire, so every later request goes out with
``x-api-key`` and nothing has to be refreshed. When no token is configured at
all, a service principal from the ``JWT_*`` environment variables supplies the
JWT.
"""
import functools
from typing import Optional, Tuple

import requests

from .. import config
from .timbr_utils import run_query

_AAD_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"


@functools.lru_cache(maxsize=1024)
def exchange_jwt(url: str, jwt: str, jwt_tenant_id: Optional[str] = None, verify_ssl: bool = True) -> str:
    """Return the Timbr token of the JWT's user. Cached per JWT, so one round trip each."""
    conn_params = {
        "url": url,
        "token": jwt,
        "ontology": "system_db",
        "verify_ssl": verify_ssl,
        "is_jwt": True,
        "jwt_tenant_id": jwt_tenant_id,
    }
    try:
        rows = run_query("SHOW TOKEN", conn_params)
    except Exception as e:
        message = str(e)
        if "401" in message or "unauthorized" in message.lower():
            raise ValueError(
                "Timbr rejected the JWT token. Check that it is valid, not expired, issued for the "
                "audience the Timbr server expects, and that jwt_tenant_id matches its tenant. "
                f"Server response: {message}"
            ) from e
        raise ValueError(f"Failed to exchange the JWT token for a Timbr token (SHOW TOKEN): {message}") from e

    token = rows[0].get("token") if rows else None
    if not token:
        raise ValueError("Failed to exchange the JWT token for a Timbr token: SHOW TOKEN returned no token")
    return token


@functools.lru_cache(maxsize=16)
def _spn_timbr_token(url: str, tenant_id: str, client_id: str, client_secret: str, scope: str, verify_ssl: bool) -> str:
    """Client-credentials JWT for the service principal, exchanged for its Timbr token."""
    response = requests.post(
        _AAD_TOKEN_URL.format(tenant_id=tenant_id),
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code != 200 or not body.get("access_token"):
        raise ValueError(
            f"Failed to get a JWT token for service principal JWT_CLIENT_ID={client_id} "
            f"(HTTP {response.status_code}): {body.get('error')}: {body.get('error_description')}"
        )
    return exchange_jwt(url, body["access_token"], tenant_id, verify_ssl)


def resolve_timbr_auth(
    url: Optional[str],
    token: Optional[str],
    is_jwt: bool,
    jwt_tenant_id: Optional[str] = None,
    verify_ssl: bool = True,
) -> Tuple[Optional[str], bool]:
    """The ``(token, is_jwt)`` to connect with: a JWT comes back as its Timbr token.

    With no token at all, a service principal from the ``JWT_*`` environment
    variables is used when one is configured; otherwise the inputs come back
    unchanged and connection validation reports what is missing.
    """
    if not url:
        return token, is_jwt
    if token:
        if is_jwt:
            return exchange_jwt(url, token, jwt_tenant_id, verify_ssl), False
        return token, is_jwt
    if not (config.jwt_client_id or config.jwt_client_secret):
        return token, is_jwt

    spn = {
        "JWT_TENANT_ID": jwt_tenant_id or config.jwt_tenant_id,
        "JWT_CLIENT_ID": config.jwt_client_id,
        "JWT_CLIENT_SECRET": config.jwt_client_secret,
        "JWT_SCOPE": config.jwt_scope,
    }
    missing = [name for name, value in spn.items() if not value]
    if missing:
        raise ValueError(f"Service principal authentication needs these environment variables: {', '.join(missing)}")
    return _spn_timbr_token(url, *spn.values(), verify_ssl), False


def resolve_conn_params_auth(
    conn_params: dict,
    url: Optional[str],
    jwt_tenant_id: Optional[str] = None,
    verify_ssl: bool = True,
) -> dict:
    """The same exchange for a JWT passed inside ``conn_params``, which overrides the chain's own token."""
    if not (conn_params.get("is_jwt") and conn_params.get("token")):
        return conn_params
    token, is_jwt = resolve_timbr_auth(
        conn_params.get("url") or url,
        conn_params["token"],
        True,
        conn_params.get("jwt_tenant_id", jwt_tenant_id),
        conn_params.get("verify_ssl", verify_ssl),
    )
    return {**conn_params, "token": token, "is_jwt": is_jwt}
