#!/usr/bin/env python3
"""
Generate the admin credentials for the leaderboard dashboard.

Run this once to produce the three environment variables the server reads.
Your password is hashed here and never stored in plain text or committed.

    python set_admin_password.py

It will prompt for your Kerberos ID and a NEW admin password (not your real
institute password), then print export lines to paste into your shell or a
.env file that is kept out of git.
"""
import getpass
import secrets

import auth


def main() -> None:
    print("Sylva admin credential setup\n")
    kerberos = input("Kerberos ID (used only as the admin username): ").strip().lower()
    if not kerberos:
        raise SystemExit("Kerberos ID cannot be empty.")

    print("\nChoose a NEW admin password for this dashboard.")
    print("Do NOT reuse your real institute/Kerberos password.\n")
    password = getpass.getpass("Admin password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        raise SystemExit("Passwords did not match.")
    if len(password) < 8:
        raise SystemExit("Please choose at least 8 characters.")

    pw_hash = auth.hash_password(password)
    session_secret = secrets.token_urlsafe(48)

    print("\nDone. Add these to your shell or an untracked .env, then start the server:\n")
    print(f'export ADMIN_KERBEROS="{kerberos}"')
    print(f'export ADMIN_PASSWORD_HASH="{pw_hash}"')
    print(f'export ADMIN_SESSION_SECRET="{session_secret}"')
    print(
        "\nThe hash and secret are safe to store; the password itself is not "
        "recoverable from them. Re-run this to rotate them."
    )


if __name__ == "__main__":
    main()
