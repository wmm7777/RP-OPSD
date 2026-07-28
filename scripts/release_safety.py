#!/usr/bin/env python3
"""Reject private paths, infrastructure identifiers, and credentials in releases."""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?:^|[\s\"'=:(])"
    r"(?:/(?!/)[a-z0-9._~-]+(?:/[^\s\"'<>]*)?"
    r"|[a-z]:\\|\\\\[a-z0-9_.-]+\\)"
)
SECRET_RE = re.compile(
    r"(?i)(?:"
    r"\bAKIA[0-9A-Z]{16}\b"
    r"|\bASIA[0-9A-Z]{16}\b"
    r"|\bhf_[A-Za-z0-9]{20,}\b"
    r"|\bsk-[A-Za-z0-9_-]{20,}\b"
    r"|-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"
    r")"
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>'\"]+")
IPV4_RE = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
INTERNAL_HOST_RE = re.compile(
    r"(?i)\b[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?"
    r"(?:\.corp|\.internal|\.lan|\.local)\b"
)
SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:"
    r"api[_-]?key"
    r"|access[_-]?(?:key|token)"
    r"|authorization"
    r"|cookie"
    r"|credential"
    r"|password"
    r"|private[_-]?key"
    r"|secret"
    r"|token"
    r")"
)
SENSITIVE_QUERY_KEYS = {
    "access_key",
    "access_token",
    "api_key",
    "credential",
    "key",
    "password",
    "secret",
    "signature",
    "token",
    "x-amz-credential",
    "x-amz-signature",
}
PUBLIC_HOST_SUFFIXES = (
    "apache.org",
    "arxiv.org",
    "doi.org",
    "github.com",
    "huggingface.co",
    "openaccess.thecvf.com",
    "paperswithcode.com",
)


def is_public_release_host(hostname: str) -> bool:
    return any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in PUBLIC_HOST_SUFFIXES
    )


def sensitive_reason(text: str) -> str | None:
    if ABSOLUTE_PATH_RE.search(text):
        return "absolute user or workspace path"
    if SECRET_RE.search(text):
        return "credential-like value"
    if EMAIL_RE.search(text):
        return "email address"
    if INTERNAL_HOST_RE.search(text):
        return "private or non-public hostname"

    for raw_url in URL_RE.findall(text):
        parsed = urlsplit(raw_url.rstrip(".,);]"))
        if parsed.scheme.lower() not in {"http", "https"}:
            return "non-public or unsupported URL scheme"
        hostname = (parsed.hostname or "").lower()
        if parsed.username or parsed.password:
            return "credential embedded in URL"
        if "." not in hostname:
            return "private or non-public hostname"
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            return "non-public IP address"
        if address is None and not is_public_release_host(hostname):
            return "URL host is not allowlisted for public release"
        query_keys = {key.lower() for key, _ in parse_qsl(parsed.query)}
        if query_keys & SENSITIVE_QUERY_KEYS:
            return "signed or credential-bearing URL"

    for candidate in IPV4_RE.findall(text):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if not address.is_global:
            return "non-public IP address"
    return None


def assert_public_value(value: Any, location: str = "value") -> None:
    if isinstance(value, str):
        reason = sensitive_reason(value)
        if reason:
            raise ValueError(f"{reason} at {location}: {value[:240]}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            assert_public_value(key_text, f"{location}.<key>")
            if (
                SENSITIVE_KEY_RE.search(key_text)
                and child not in (None, "", False, [], {})
            ):
                raise ValueError(f"sensitive field at {location}.{key_text}")
            assert_public_value(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_public_value(child, f"{location}[{index}]")


def assert_public_text(path: Path, location: str | None = None) -> None:
    assert_public_value(
        path.read_text(encoding="utf-8"),
        location or path.name,
    )
