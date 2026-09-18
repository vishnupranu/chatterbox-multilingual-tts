"""
auth.py
Authentication & Session Management for Khyathi.Sri AI Platform.
Provides secure user registration, password hashing (PBKDF2-HMAC-SHA256),
token-based session handling, user profile management, and role-based presets
(Kids & Family, School/College Student, Professional, PG Researcher).
"""

import os
import sqlite3
import hashlib
import secrets
import datetime
from typing import Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes users and sessions tables and seeds demo accounts."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()

    # Seed demo accounts if empty
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        demo_users = [
            ("demo_researcher", "researcher@khyathi.sri", "researcher123", "Dr. Aryan Sharma", "PG Researcher"),
            ("demo_student", "student@khyathi.sri", "student123", "Priya Patel", "College Student"),
            ("demo_educator", "educator@khyathi.sri", "educator123", "Khyathi Sri", "Early Educator / Parent"),
            ("demo_work", "office@khyathi.sri", "office123", "Vikram Rao", "Business Professional"),
        ]
        for uid, email, pwd, name, role in demo_users:
            salt = secrets.token_hex(16)
            pwd_hash = hash_password(pwd, salt)
            cursor.execute(
                "INSERT INTO users (id, email, password_hash, salt, name, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, email.lower(), pwd_hash, salt, name, role, datetime.datetime.utcnow().isoformat())
            )
        conn.commit()
    conn.close()


def hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 password hashing."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()


def register_user(email: str, password: str, name: str, role: str = "Student") -> Dict[str, Any]:
    """Registers a new user and generates a session token."""
    init_db()
    email = email.strip().lower()
    name = name.strip()
    if not email or "@" not in email:
        return {"success": False, "error": "Valid email address is required."}
    if not password or len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters long."}
    if not name:
        name = email.split("@")[0].capitalize()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return {"success": False, "error": "An account with this email already exists."}

    user_id = "usr_" + secrets.token_hex(8)
    salt = secrets.token_hex(16)
    pwd_hash = hash_password(password, salt)
    created_at = datetime.datetime.utcnow().isoformat()

    cursor.execute(
        "INSERT INTO users (id, email, password_hash, salt, name, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, email, pwd_hash, salt, name, role, created_at)
    )

    token = "kstk_" + secrets.token_hex(24)
    cursor.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)", (token, user_id, created_at))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "token": token,
        "user": {
            "id": user_id,
            "email": email,
            "name": name,
            "role": role,
            "created_at": created_at
        }
    }


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticates user and returns active session token."""
    init_db()
    email = email.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password_hash, salt, name, role, created_at FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {"success": False, "error": "Invalid email or password."}

    user_id, u_email, pwd_hash, salt, name, role, created_at = (
        row["id"], row["email"], row["password_hash"], row["salt"], row["name"], row["role"], row["created_at"]
    )

    if hash_password(password, salt) != pwd_hash:
        conn.close()
        return {"success": False, "error": "Invalid email or password."}

    token = "kstk_" + secrets.token_hex(24)
    now = datetime.datetime.utcnow().isoformat()
    cursor.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)", (token, user_id, now))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "token": token,
        "user": {
            "id": user_id,
            "email": u_email,
            "name": name,
            "role": role,
            "created_at": created_at
        }
    }


def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """Returns user record for a valid session token."""
    if not token:
        return None
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.email, u.name, u.role, u.created_at
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token = ?
    """, (token,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "role": row["role"],
        "created_at": row["created_at"]
    }


def logout_user(token: str) -> bool:
    """Invalidates session token."""
    if not token:
        return False
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# Ensure database is prepped on module import
init_db()
