"""
auth.py
Authentication & Role-Based Access Control (RBAC) for Khyathi.Sri AI Platform.
Enforces strict Super Admin (Me / Root) and Admin permission mandates alongside
User role tiers (PG Researcher, College Student, Business Professional, Early Educator).
"""

import os
import sqlite3
import hashlib
import secrets
import datetime
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

# Standardized Role Constants
ROLE_SUPER_ADMIN = "Super Admin"
ROLE_ADMIN = "Admin"
ROLE_RESEARCHER = "PG Researcher"
ROLE_STUDENT = "College Student"
ROLE_PROFESSIONAL = "Business Professional"
ROLE_EDUCATOR = "Early Educator"

ALL_ROLES = [
    ROLE_SUPER_ADMIN,
    ROLE_ADMIN,
    ROLE_RESEARCHER,
    ROLE_STUDENT,
    ROLE_PROFESSIONAL,
    ROLE_EDUCATOR
]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 password hashing."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()


def init_db():
    """Initializes users and sessions tables and seeds Super Admin, Admin, and demo accounts."""
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

    # Seed or ensure Super Admin and Admin accounts exist
    seed_accounts = [
        ("usr_superadmin", "superadmin@khyathi.sri", "superadmin123", "Super Admin (Owner)", ROLE_SUPER_ADMIN),
        ("usr_admin", "admin@khyathi.sri", "admin123", "Platform Admin", ROLE_ADMIN),
        ("usr_researcher", "researcher@khyathi.sri", "researcher123", "Dr. Aryan Sharma", ROLE_RESEARCHER),
        ("usr_student", "student@khyathi.sri", "student123", "Priya Patel", ROLE_STUDENT),
        ("usr_educator", "educator@khyathi.sri", "educator123", "Khyathi Sri", ROLE_EDUCATOR),
        ("usr_office", "office@khyathi.sri", "office123", "Vikram Rao", ROLE_PROFESSIONAL),
    ]

    for uid, email, pwd, name, role in seed_accounts:
        cursor.execute("SELECT id, role FROM users WHERE email = ?", (email.lower(),))
        existing = cursor.fetchone()
        if not existing:
            salt = secrets.token_hex(16)
            pwd_hash = hash_password(pwd, salt)
            cursor.execute(
                "INSERT INTO users (id, email, password_hash, salt, name, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, email.lower(), pwd_hash, salt, name, role, datetime.datetime.utcnow().isoformat())
            )
        else:
            # Ensure correct role
            if existing["role"] != role:
                cursor.execute("UPDATE users SET role = ? WHERE id = ?", (role, existing["id"]))

    conn.commit()
    conn.close()


def register_user(email: str, password: str, name: str, role: str = ROLE_STUDENT) -> Dict[str, Any]:
    """Registers a new user. Super Admin role cannot be self-registered."""
    init_db()
    email = email.strip().lower()
    name = name.strip()
    if not email or "@" not in email:
        return {"success": False, "error": "Valid email address is required."}
    if not password or len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters long."}
    if not name:
        name = email.split("@")[0].capitalize()

    # Restrict direct registration of Super Admin unless granted
    if role == ROLE_SUPER_ADMIN:
        role = ROLE_RESEARCHER

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
            "is_super_admin": (role == ROLE_SUPER_ADMIN),
            "is_admin": (role in (ROLE_SUPER_ADMIN, ROLE_ADMIN)),
            "created_at": created_at
        }
    }


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticates user and returns session token with permission flags."""
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

    is_super = (role == ROLE_SUPER_ADMIN)
    is_adm = (role in (ROLE_SUPER_ADMIN, ROLE_ADMIN))

    return {
        "success": True,
        "token": token,
        "user": {
            "id": user_id,
            "email": u_email,
            "name": name,
            "role": role,
            "is_super_admin": is_super,
            "is_admin": is_adm,
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

    role = row["role"]
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "role": role,
        "is_super_admin": (role == ROLE_SUPER_ADMIN),
        "is_admin": (role in (ROLE_SUPER_ADMIN, ROLE_ADMIN)),
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


# =====================================================================
# PERMISSION & ADMIN CONTROLS (MANDATED FOR SUPER ADMIN & ADMIN)
# =====================================================================

def verify_super_admin(token: str) -> bool:
    """Returns True only if the provided token belongs to a Super Admin."""
    user = get_current_user(token)
    return bool(user and user.get("is_super_admin"))


def verify_admin(token: str) -> bool:
    """Returns True if the provided token belongs to an Admin or Super Admin."""
    user = get_current_user(token)
    return bool(user and user.get("is_admin"))


def list_all_users(token: str) -> Optional[List[Dict[str, Any]]]:
    """Super Admin only: Returns all registered platform users."""
    if not verify_super_admin(token):
        return None
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, name, role, created_at FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "email": r["email"],
            "name": r["name"],
            "role": r["role"],
            "created_at": r["created_at"]
        }
        for r in rows
    ]


def get_admin_system_stats(token: str) -> Optional[Dict[str, Any]]:
    """Admin & Super Admin: Returns platform health and permission metrics."""
    if not verify_admin(token):
        return None
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sessions")
    active_sessions = cursor.fetchone()[0]
    cursor.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
    role_breakdown = {r["role"]: r["count"] for r in cursor.fetchall()}
    conn.close()

    return {
        "total_users": total_users,
        "active_sessions": active_sessions,
        "role_breakdown": role_breakdown,
        "hardware_engine": "Apple Silicon MPS Neural Core (24kHz HiFi)",
        "security_partner": "Key Secure Foundation Cloud",
        "permission_tier": "SUPER_ADMIN_ROOT" if verify_super_admin(token) else "ADMIN_PLATFORM"
    }


# Initialize DB on module load
init_db()
