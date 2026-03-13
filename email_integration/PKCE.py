"""
In-memory PKCE (Proof Key for Code Exchange) store for OAuth2 authorization code flows.

PKCE prevents authorization code interception attacks by binding a one-time code
verifier to the OAuth state parameter.  The flow is:

  1. Generate a (verifier, challenge) pair with make_pkce_pair().
  2. Store the verifier keyed by the state value with pkce_put().
  3. Include the challenge in the OAuth authorization URL sent to the provider.
  4. On callback, retrieve and consume the verifier with pkce_pop(state).
     The verifier is then sent to the token endpoint to prove authenticity.

Entries expire after a configurable TTL (default 10 minutes) to prevent unbounded
memory growth from abandoned login attempts.
"""

# top-level
import base64, hashlib, time
from typing import Dict, Tuple, Optional
import os

# In-memory store mapping OAuth state value -> (verifier, expiry_timestamp).
# state -> (verifier, exp_ts)
_pkce_store: Dict[str, Tuple[str, float]] = {}


def _b64url(b: bytes) -> str:
    """Encode bytes as URL-safe Base64 with padding stripped (RFC 7636 requirement)."""
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_pkce_pair():
    """Generate a fresh PKCE verifier and its SHA-256 code challenge.

    Returns:
        (verifier, challenge): Both are URL-safe Base64 strings.
        The verifier is kept secret; the challenge is sent to the OAuth provider.
    """
    verifier = _b64url(os.urandom(32))                          # 256-bit random verifier
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())  # S256 challenge
    return verifier, challenge


def pkce_put(state: str, verifier: str, ttl_sec: int = 600):
    """Store a PKCE verifier under the given OAuth state value with a TTL.

    Args:
        state:    The opaque state string sent to the OAuth provider.
        verifier: The PKCE code verifier to associate with this state.
        ttl_sec:  Seconds before this entry expires (default 600 = 10 minutes).
    """
    _pkce_store[state] = (verifier, time.time() + ttl_sec)


def pkce_pop(state: str) -> Optional[str]:
    """Retrieve and remove the PKCE verifier for the given state.

    This is a destructive read — the entry is deleted from the store on first access
    to prevent replay attacks.

    Returns:
        The verifier string, or None if the state is unknown or has expired.
    """
    item = _pkce_store.pop(state, None)
    if not item:
        return None
    verifier, exp = item
    # Reject entries that have passed their TTL.
    if time.time() > exp:
        return None
    return verifier
