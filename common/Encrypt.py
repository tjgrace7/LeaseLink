"""
AES-256-GCM encryption and decryption helpers for OAuth access/refresh tokens.

The symmetric key is loaded once at import time from the ENCRYPTION_KEY environment
variable (expected as a hex-encoded 32-byte value).  Each encrypted token is stored
as a Base64 string that prefixes the 12-byte random nonce followed by the GCM
ciphertext+tag, making it self-contained for decryption.
"""

import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Load the AES-256 key once from env; fail fast if the variable is missing.
Raw_Key = bytes.fromhex(os.environ["ENCRYPTION_KEY"])


def encrypt_token(token: str) -> str:
    """Encrypt a plaintext token string using AES-256-GCM.

    A fresh 12-byte random nonce is generated for every call, which is prepended
    to the ciphertext before Base64 encoding.  The returned string is safe to
    store in the database.
    """
    aesgcm = AESGCM(Raw_Key)
    nonce = os.urandom(12)                                    # 96-bit nonce, unique per encryption
    ciphertext = aesgcm.encrypt(nonce, token.encode('utf-8'), None)
    # Concatenate nonce + ciphertext+tag and Base64-encode for DB storage.
    return base64.b64encode(nonce + ciphertext).decode('utf-8')


def decrypt_token(enc_b64: str) -> str:
    """Decrypt a Base64-encoded AES-256-GCM token previously produced by encrypt_token.

    Splits the decoded bytes back into the 12-byte nonce and the ciphertext+tag,
    then decrypts and returns the original plaintext string.
    """
    #Decrypt Tokens
    data = base64.b64decode(enc_b64)
    nonce, ciphertext = data[:12], data[12:]    # first 12 bytes are the nonce
    aesgcm = AESGCM(Raw_Key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode('utf-8')