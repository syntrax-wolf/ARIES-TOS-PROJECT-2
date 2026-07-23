"""
Admin authentication for the leaderboard dashboard.

Deliberately does NOT touch real institute credentials. The admin's Kerberos
ID is used only as a username (it is a public identifier); the password is a
separate one the admin sets, stored as a salted PBKDF2 hash in an environment
variable and never in the repository.

Config (all via environment, see set_admin_password.py to generate them):
    ADMIN_KERBEROS       the admin's Kerberos id, e.g. "cs1210456"
    ADMIN_PASSWORD_HASH  "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>"
    ADMIN_SESSION_SECRET random secret used to sign session cookies

If the config is absent the admin surface is simply disabled (login always
fails) rather than falling back to a default password.
"""
import hashlib
import hmac
import os
import secrets
import time

SESSION_COOKIE = "sylva_admin"
SESSION_TTL_SECONDS = 8 * 3600  # a working session; re-login after that

PBKDF2_ITERATIONS = 240_000
_ALGO = "pbkdf2_sha256"

# Simple in-process brute-force brake: too many bad tries and the account is
# briefly locked. Fine for a single-admin tool on one server.
_MAX_ATTEMPTS = 8
_LOCKOUT_SECONDS = 300
_attempts: dict[str, list[float]] = {}


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{_ALGO}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = encoded.split("$")
        if algo != _ALGO:
            return False
        expected = bytes.fromhex(hash_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(actual, expected)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _config() -> tuple[str, str, str] | None:
    kerberos = os.environ.get("ADMIN_KERBEROS", "").strip().lower()
    pw_hash = os.environ.get("ADMIN_PASSWORD_HASH", "").strip()
    secret = os.environ.get("ADMIN_SESSION_SECRET", "").strip()
    if kerberos and pw_hash and secret:
        return kerberos, pw_hash, secret
    return None


def is_configured() -> bool:
    return _config() is not None


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def _locked_out(kerberos: str) -> bool:
    now = time.time()
    recent = [t for t in _attempts.get(kerberos, []) if now - t < _LOCKOUT_SECONDS]
    _attempts[kerberos] = recent
    return len(recent) >= _MAX_ATTEMPTS


def _record_failure(kerberos: str) -> None:
    _attempts.setdefault(kerberos, []).append(time.time())


class LoginError(Exception):
    """Message is safe to show the admin."""


def login(kerberos: str, password: str) -> str:
    """Validate credentials, return a signed session token, or raise LoginError."""
    config = _config()
    if config is None:
        raise LoginError("Admin access is not configured on this server.")

    kerberos = (kerberos or "").strip().lower()
    admin_id, pw_hash, secret = config

    if _locked_out(kerberos):
        raise LoginError("Too many attempts. Try again in a few minutes.")

    # Verify the id and the password independently, then AND the results, so the
    # response time does not reveal which half was wrong.
    id_ok = hmac.compare_digest(kerberos, admin_id)
    pw_ok = verify_password(password or "", pw_hash)
    if not (id_ok and pw_ok):
        _record_failure(kerberos)
        raise LoginError("Incorrect Kerberos ID or password.")

    _attempts.pop(kerberos, None)
    return _issue_token(secret)


# ---------------------------------------------------------------------------
# Session tokens — stateless, HMAC-signed "<expiry>.<sig>"
# ---------------------------------------------------------------------------

def _issue_token(secret: str) -> str:
    expiry = int(time.time()) + SESSION_TTL_SECONDS
    payload = str(expiry)
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def valid_session(token: str | None) -> bool:
    config = _config()
    if config is None or not token:
        return False
    secret = config[2]
    try:
        payload, sig = token.rsplit(".", 1)
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        return int(payload) > int(time.time())
    except (ValueError, AttributeError):
        return False
