#!/usr/bin/env python3
"""
Claude Code PreToolUse Hook — Credential & Secret Guard
=========================================================
Intercepts every tool call BEFORE execution. Returns exit code 2 to block.
This is defense-in-depth: deny rules in settings.json are the first layer,
but this hook catches what deny rules might miss (especially Bash bypasses).

Deploy: .claude/hooks/guard_secrets.py
Config: Add to project .claude/settings.json (NOT user settings)

Compatible with Python 3.9+ (macOS CommandLineTools default)
"""

import json
import sys
import re
from typing import Optional

# ── Patterns that indicate credential/secret file access ──────────────
SENSITIVE_FILE_PATTERNS = [
    # Environment files
    r'\.env($|\.)',
    r'\.env\.local',
    r'\.env\.production',
    r'\.env\.staging',
    r'\.env\.development', r'\.envs/',

    # Keys and certificates
    r'\.pem$', r'\.key$', r'\.p12$', r'\.pfx$', r'\.jks$',
    r'\.keystore$', r'\.crt$', r'\.cer$', r'\.der$',
    r'id_rsa', r'id_ed25519', r'id_ecdsa', r'id_dsa',
    r'known_hosts',

    # AWS
    r'\.aws/credentials', r'\.aws/config',
    r'aws.credentials', r'aws-credentials',

    # GCP
    r'\.config/gcloud', r'service[-_]account.*\.json',
    r'gcp[-_]credentials', r'application_default_credentials',

    # SSH / GPG
    r'\.ssh/', r'\.gnupg/',

    # Terraform / IaC
    r'terraform\.tfstate', r'\.tfvars$',

    # Ansible vaults
    r'vault.*\.ya?ml$', r'group_vars/.*/vault', r'host_vars/.*/vault',

    # Kubernetes / Docker
    r'\.kube/config', r'\.docker/config\.json', r'kubeconfig',

    # Application secrets
    r'secrets/', r'secret[s]?\.', r'credentials\.(json|yml|yaml)',
    r'\.npmrc$', r'\.pypirc$', r'\.netrc$', r'\.htpasswd$', r'\.pgpass$',
    r'password', r'token.*\.json',

    # Django secrets
    r'local_settings\.py', r'settings_local\.py', r'secret_key',

    # Pulumi
    r'Pulumi\..*\.yaml', r'\.pulumi/',

    # AWS legacy tools
    r'\.boto$', r'\.s3cfg$',

    # Azure
    r'\.azure/', r'azure[-_]credentials', r'servicePrincipal.*\.json',

    # Databases (PostgreSQL, MySQL, MongoDB, Redis)
    r'\.my\.cnf$', r'\.mylogin\.cnf$',
    r'\.mongorc\.js$', r'\.mongoshrc\.js$',
    r'\.dbshell$', r'\.psql_history$', r'\.mysql_history$',
    r'\.rediscli_history$', r'\.sqlite_history$',

    # HashiCorp Vault
    r'\.vault-token$', r'vault\.json$',

    # Firebase
    r'firebase.*adminsdk.*\.json', r'firebaseServiceAccount.*\.json',

    # Python pip private index
    r'pip\.conf$',

    # Frontend deployment
    r'\.vercel', r'\.netlify',

    # Docker legacy
    r'\.dockercfg$',

    # Ansible vault password
    r'\.vault_pass',

    # VPN configs
    r'\.ovpn$', r'\.wg$', r'vpn.*\.conf$',
    r'openvpn/', r'wireguard/', r'\.tblk/',

    # ArgoCD
    r'\.argocd/', r'argocd-secret', r'argocd-cm\.yaml',

    # Helm
    r'\.helm/', r'\.config/helm/', r'helm/repositories\.yaml',
    r'values[-.]secret', r'values\.secret\.yaml',

    # Kubernetes additional
    r'\.minikube/',

    # Database configs
    r'mongod\.conf', r'pg_hba\.conf', r'redis\.conf',
    r'\.pgservice\.conf',

    # Firebase / Flutter
    r'google-services\.json', r'GoogleService-Info\.plist',
    r'android/key\.properties', r'\.pub-credentials\.json',

    # Terraform / CDK
    r'\.terraform/', r'cdk\.out/',
]

# ── Bash commands that leak environment or exfiltrate data ────────────
DANGEROUS_BASH_PATTERNS = [
    # Network exfiltration
    r'\bcurl\b', r'\bwget\b', r'\bnc\b', r'\bncat\b', r'\bnetcat\b',
    r'\btelnet\b', r'\bftp\b', r'\bsftp\b', r'\bssh\b', r'\bscp\b',
    r'\brsync\b',

    # Environment variable dumping
    r'\benv\b', r'\bprintenv\b', r'\bset\b(?!\s+-)',
    r'\bexport\b',
    r'echo\s+\$AWS', r'echo\s+\$GCP', r'echo\s+\$GOOGLE',
    r'echo\s+\$SECRET', r'echo\s+\$TOKEN', r'echo\s+\$PASSWORD',
    r'echo\s+\$API_KEY', r'echo\s+\$ANTHROPIC', r'echo\s+\$DB_',
    r'echo\s+\$AZURE', r'echo\s+\$OPENAI', r'echo\s+\$DJANGO',
    r'echo\s+\$PULUMI', r'echo\s+\$NPM_TOKEN',
    r'echo\s+\$HOME/\.aws', r'echo\s+\$HOME/\.ssh',

    # Privilege escalation
    r'\bsudo\b', r'\bsu\s', r'\bchmod\s+777\b',
    r'\bchown\b', r'\bchgrp\b',

    # Destructive operations
    r'\brm\s+-rf\s+/', r'\bdd\s+', r'\bmkfs\b', r'\bfdisk\b',

    # Data encoding (exfiltration prep)
    r'\bbase64\b.*<', r'\bxxd\b',
    r'\bopenssl\b',

    # Git exfiltration
    r'git\s+remote\s+add', r'git\s+remote\s+set-url',
    r'git\s+push\s+--force', r'git\s+push\s+-f\b',
    r'git\s+commit\s+--no-verify', r'git\s+commit\s+-n\b',
    r'git\s+fast-export', r'git\s+bundle\s+create',
    r'git\s+send-pack', r'git\s+submodule\s+add',

    # Process/network inspection (info gathering)
    r'\bps\s+aux\b', r'\bstrace\b', r'\bltrace\b',
    r'\blsof\s+-i\b', r'\bnetstat\b', r'\bss\s+-tulnp\b',

    # Package installation (supply chain risk)
    r'\bpip3?\s+install\b', r'\bnpm\s+install\b',
    r'\byarn\s+add\b', r'\bpnpm\s+add\b',
    r'\bgem\s+install\b',

    # Bash variable expansion bypass attempts
    r'\$\{.*curl.*\}', r'\$\{.*wget.*\}',
    r'eval\s+', r'bash\s+-c\s+',
]


def check_file_path(path: str) -> Optional[str]:
    """Check if a file path matches sensitive patterns."""
    for pattern in SENSITIVE_FILE_PATTERNS:
        if re.search(pattern, path, re.IGNORECASE):
            return pattern
    return None


def check_bash_command(command: str) -> Optional[str]:
    """Check if a bash command matches dangerous patterns."""
    for pattern in DANGEROUS_BASH_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return pattern
    return None


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        # Can't parse input — allow (fail open for usability)
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # ── Check Read / Edit / Write tools ───────────────────────────
    if tool_name in ("Read", "Edit", "Write", "MultiEdit"):
        file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
        matched = check_file_path(file_path)
        if matched:
            msg = (
                f"BLOCKED by guard_secrets hook: "
                f"{tool_name} on '{file_path}' matches sensitive pattern '{matched}'. "
                f"Credentials and secrets must not be read or modified by Claude Code."
            )
            print(msg, file=sys.stderr)
            sys.exit(2)

    # ── Check Bash tool ───────────────────────────────────────────
    if tool_name == "Bash":
        command = tool_input.get("command", "")

        # Check for cat/less/head/tail of sensitive files
        file_match = check_file_path(command)
        if file_match:
            msg = (
                f"BLOCKED by guard_secrets hook: "
                f"Bash command references sensitive file pattern '{file_match}'. "
                f"Command: {command[:120]}..."
            )
            print(msg, file=sys.stderr)
            sys.exit(2)

        # Check for dangerous command patterns
        cmd_match = check_bash_command(command)
        if cmd_match:
            msg = (
                f"BLOCKED by guard_secrets hook: "
                f"Bash command matches dangerous pattern '{cmd_match}'. "
                f"Command: {command[:120]}..."
            )
            print(msg, file=sys.stderr)
            sys.exit(2)

    # ── Check WebFetch tool ───────────────────────────────────────
    if tool_name == "WebFetch":
        msg = "BLOCKED by guard_secrets hook: WebFetch is disabled by security policy."
        print(msg, file=sys.stderr)
        sys.exit(2)

    # ── All clear ─────────────────────────────────────────────────
    sys.exit(0)


if __name__ == "__main__":
    main()
