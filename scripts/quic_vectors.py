#!/usr/bin/env python3
"""Compute the RFC 9001 Appendix A QUIC Initial key test vectors from scratch,
so we can assert against the exact bytes in the Rust unit test."""
import hashlib, hmac

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out = b""
    t = b""
    counter = 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:length]

def hkdf_expand_label(secret: bytes, label: str, context: bytes, length: int) -> bytes:
    full = b"tls13 " + label.encode()
    info = length.to_bytes(2, "big") + bytes([len(full)]) + full + bytes([len(context)]) + context
    return hkdf_expand(secret, info, length)

QUIC_V1_INITIAL_SALT = bytes.fromhex("38762cf7f55934b34d179ae6a4c80cadccbb7f0a")
client_dcid = bytes.fromhex("8394c8f03e515708")

initial_secret = hkdf_extract(QUIC_V1_INITIAL_SALT, client_dcid)
client_initial_secret = hkdf_expand_label(initial_secret, "client in", b"", 32)
server_initial_secret = hkdf_expand_label(initial_secret, "server in", b"", 32)
client_key = hkdf_expand_label(client_initial_secret, "quic key", b"", 16)
client_iv = hkdf_expand_label(client_initial_secret, "quic iv", b"", 12)
client_hp = hkdf_expand_label(client_initial_secret, "quic hp", b"", 16)

print("initial_secret        =", initial_secret.hex())
print("client_initial_secret =", client_initial_secret.hex())
print("server_initial_secret =", server_initial_secret.hex())
print("client_key            =", client_key.hex())
print("client_iv             =", client_iv.hex())
print("client_hp             =", client_hp.hex())
