# top-level
import base64, hashlib, time
from typing import Dict, Tuple, Optional
import os

# state -> (verifier, exp_ts)
_pkce_store: Dict[str, Tuple[str, float]] = {}

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def make_pkce_pair():
    verifier = _b64url(os.urandom(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge

def pkce_put(state: str, verifier: str, ttl_sec: int = 600):
    _pkce_store[state] = (verifier, time.time() + ttl_sec)

def pkce_pop(state: str) -> Optional[str]:
    item = _pkce_store.pop(state, None)
    if not item:
        return None
    verifier, exp = item
    if time.time() > exp:
        return None
    return verifier
