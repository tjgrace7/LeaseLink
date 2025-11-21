import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

Raw_Key = bytes.fromhex(os.environ["ENCRYPTION_KEY"])

def encrypt_token(token: str) -> str:
    aesgcm = AESGCM(Raw_Key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, token.encode('utf-8'), None)
    return base64.b64encode(nonce+ciphertext).decode('utf-8')
def decrypt_token(enc_b64: str) -> str:
    #Decrypt Tokens
    data = base64.b64decode(enc_b64)
    nonce, ciphertext = data[:12], data[12:]
    aesgcm = AESGCM(Raw_Key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode('utf-8')