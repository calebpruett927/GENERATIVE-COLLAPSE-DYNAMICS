"""
UMCP Security Domain - Tier-2 Overlay: Reputation Analyzer

Analyzes reputation of URLs, domains, IPs, and file hashes.
This is a DIAGNOSTIC overlay - it reads external data sources but results
are Tier-2 diagnostics that cannot alter Tier-1 invariants.

Tier-2 rules:
    - May call external APIs (VirusTotal, etc.) for reputation data
    - Results are DIAGNOSTIC - they inform but don't define trust
    - Reputation feeds into Tier-0 signals which then flow through Tier-1
    - No back-edges: external data cannot retroactively change frozen policy

Reputation Sources:
    - Local blocklists (frozen at Tier-0)
    - External APIs (optional, not required for core validation)
    - Domain age and registration
    - SSL certificate analysis
    - Historical behavior patterns
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse


class ReputationType(Enum):
    """Reputation classification types."""

    TRUSTED = "TRUSTED"
    NEUTRAL = "NEUTRAL"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"
    UNKNOWN = "UNKNOWN"


@dataclass
class ReputationResult:
    """Reputation analysis result."""

    reputation_type: ReputationType
    score: float  # 0.0 (malicious) to 1.0 (trusted)
    sources: list[str]
    indicators: list[str]
    details: dict[str, Any]


# Suspicious TLDs (from threat_patterns.v1.yaml)
SUSPICIOUS_TLDS = [".xyz", ".top", ".click", ".loan", ".work", ".gq", ".ml", ".cf", ".tk"]

# Suspicious keywords in URLs
SUSPICIOUS_KEYWORDS = ["login", "verify", "suspend", "urgent", "account", "secure", "update", "confirm"]

# Known trusted domains (simplified example)
TRUSTED_DOMAINS = [
    "google.com",
    "github.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "cloudflare.com",
    "stackoverflow.com",
]

# Homoglyph mappings (characters that look similar)
HOMOGLYPHS = {
    "0": "o",
    "o": "0",
    "1": "l",
    "l": "1",
    "i": "1",
    "3": "e",
    "e": "3",
    "4": "a",
    "a": "4",
    "5": "s",
    "s": "5",
    "7": "t",
    "t": "7",
    "8": "b",
    "b": "8",
    "@": "a",
}


def detect_homoglyphs(domain: str, known_brands: list[str] | None = None) -> list[str]:
    """
    Detect potential homoglyph attacks in domain names.

    Args:
        domain: Domain to analyze
        known_brands: List of brand names to check against

    Returns:
        List of detected homoglyph indicators
    """
    if known_brands is None:
        known_brands = ["google", "amazon", "apple", "microsoft", "paypal", "netflix", "facebook"]

    indicators = []
    domain_lower = domain.lower()

    for brand in known_brands:
        # Check for exact match (not a homoglyph)
        if brand in domain_lower:
            continue

        # Check for homoglyph variations
        for char, replacement in HOMOGLYPHS.items():
            variant = brand.replace(char, replacement)
            if variant != brand and variant in domain_lower:
                indicators.append(f"homoglyph:{brand}→{variant}")

    return indicators


def analyze_url_structure(url: str) -> dict[str, Any]:
    """
    Analyze URL structure for suspicious patterns.

    Args:
        url: URL to analyze

    Returns:
        Analysis results with indicators
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return {"error": "Invalid URL", "indicators": ["malformed_url"]}

    indicators: list[str] = []
    details: dict[str, Any] = {
        "scheme": parsed.scheme,
        "domain": parsed.netloc,
        "path": parsed.path,
        "query": parsed.query,
    }

    # Check scheme
    if parsed.scheme not in ["https", "http"]:
        indicators.append("unusual_scheme")
    if parsed.scheme == "http":
        indicators.append("no_ssl")

    # Check for IP address instead of domain
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    if re.match(ip_pattern, parsed.netloc.split(":")[0]):
        indicators.append("ip_address_url")

    # Check for suspicious TLD
    for tld in SUSPICIOUS_TLDS:
        if parsed.netloc.endswith(tld):
            indicators.append(f"suspicious_tld:{tld}")
            break

    # Check for suspicious keywords
    url_lower = url.lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in url_lower:
            indicators.append(f"suspicious_keyword:{keyword}")

    # Check for excessive subdomains
    subdomain_count = parsed.netloc.count(".")
    if subdomain_count > 3:
        indicators.append(f"excessive_subdomains:{subdomain_count}")

    # Check for homoglyphs
    homoglyph_hits = detect_homoglyphs(parsed.netloc)
    indicators.extend(homoglyph_hits)

    # Check URL length
    if len(url) > 200:
        indicators.append("excessive_length")

    details["indicators"] = indicators
    details["indicator_count"] = len(indicators)

    return details


def analyze_url_reputation(
    url: str, local_blocklist: list[str] | None = None, use_external_api: bool = False
) -> ReputationResult:
    """
    Analyze URL reputation.

    Args:
        url: URL to analyze
        local_blocklist: Optional local blocklist of malicious URLs/domains
        use_external_api: Whether to query external APIs (disabled by default)

    Returns:
        ReputationResult with score and indicators
    """
    sources = ["local_analysis"]
    indicators = []
    details = {}

    # Parse URL
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return ReputationResult(
            reputation_type=ReputationType.MALICIOUS,
            score=0.0,
            sources=["parse_error"],
            indicators=["malformed_url"],
            details={"error": "Could not parse URL"},
        )

    # Check local blocklist
    if local_blocklist:
        for blocked in local_blocklist:
            if blocked in url or blocked in domain:
                return ReputationResult(
                    reputation_type=ReputationType.MALICIOUS,
                    score=0.0,
                    sources=["local_blocklist"],
                    indicators=[f"blocklist_match:{blocked}"],
                    details={"blocked_pattern": blocked},
                )

    # Check trusted domains
    for trusted in TRUSTED_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return ReputationResult(
                reputation_type=ReputationType.TRUSTED,
                score=0.95,
                sources=["trusted_list"],
                indicators=["known_trusted_domain"],
                details={"trusted_domain": trusted},
            )

    # Analyze URL structure
    structure = analyze_url_structure(url)
    indicators.extend(structure.get("indicators", []))
    details["structure"] = structure

    # Calculate score based on indicators
    base_score = 0.5  # Neutral starting point

    # Deductions for suspicious indicators
    indicator_penalties = {
        "suspicious_tld": -0.2,
        "suspicious_keyword": -0.1,
        "homoglyph": -0.3,
        "ip_address_url": -0.15,
        "no_ssl": -0.1,
        "excessive_subdomains": -0.1,
        "excessive_length": -0.05,
        "malformed_url": -0.5,
    }

    for indicator in indicators:
        for pattern, penalty in indicator_penalties.items():
            if pattern in indicator:
                base_score += penalty
                break

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, base_score))

    # Determine reputation type
    if score >= 0.8:
        rep_type = ReputationType.TRUSTED
    elif score >= 0.5:
        rep_type = ReputationType.NEUTRAL
    elif score >= 0.2:
        rep_type = ReputationType.SUSPICIOUS
    else:
        rep_type = ReputationType.MALICIOUS

    return ReputationResult(
        reputation_type=rep_type, score=score, sources=sources, indicators=indicators, details=details
    )


def analyze_file_hash(
    file_hash: str,
    hash_type: str = "sha256",
    local_blocklist: list[str] | None = None,
    local_allowlist: list[str] | None = None,
) -> ReputationResult:
    """
    Analyze file hash reputation.

    Args:
        file_hash: Hash to analyze
        hash_type: Type of hash (sha256, md5, sha1)
        local_blocklist: Known malicious hashes
        local_allowlist: Known trusted hashes

    Returns:
        ReputationResult
    """
    file_hash = file_hash.lower().strip()

    # Validate hash format
    hash_lengths = {"md5": 32, "sha1": 40, "sha256": 64}
    expected_len = hash_lengths.get(hash_type)

    if expected_len and len(file_hash) != expected_len:
        return ReputationResult(
            reputation_type=ReputationType.UNKNOWN,
            score=0.5,
            sources=["validation_error"],
            indicators=[f"invalid_hash_length:{len(file_hash)}"],
            details={"expected_length": expected_len, "actual_length": len(file_hash)},
        )

    # Check allowlist
    if local_allowlist and file_hash in local_allowlist:
        return ReputationResult(
            reputation_type=ReputationType.TRUSTED,
            score=0.95,
            sources=["local_allowlist"],
            indicators=["known_trusted_hash"],
            details={"hash": file_hash, "hash_type": hash_type},
        )

    # Check blocklist
    if local_blocklist and file_hash in local_blocklist:
        return ReputationResult(
            reputation_type=ReputationType.MALICIOUS,
            score=0.0,
            sources=["local_blocklist"],
            indicators=["known_malicious_hash"],
            details={"hash": file_hash, "hash_type": hash_type},
        )

    # Unknown hash - neutral reputation
    return ReputationResult(
        reputation_type=ReputationType.UNKNOWN,
        score=0.5,
        sources=["no_match"],
        indicators=["hash_not_in_database"],
        details={"hash": file_hash, "hash_type": hash_type},
    )


def analyze_ip_reputation(ip_address: str, local_blocklist: list[str] | None = None) -> ReputationResult:
    """
    Analyze IP address reputation.

    Args:
        ip_address: IP to analyze
        local_blocklist: Known malicious IPs

    Returns:
        ReputationResult
    """
    indicators = []

    # Check if private IP
    private_ranges = [
        ("10.", "10.255.255.255"),
        ("172.16.", "172.31.255.255"),
        ("192.168.", "192.168.255.255"),
        ("127.", "127.255.255.255"),
    ]

    is_private = any(ip_address.startswith(prefix) for prefix, _ in private_ranges)

    if is_private:
        indicators.append("private_ip")
        return ReputationResult(
            reputation_type=ReputationType.NEUTRAL,
            score=0.6,
            sources=["local_analysis"],
            indicators=indicators,
            details={"ip": ip_address, "is_private": True},
        )

    # Check blocklist
    if local_blocklist and ip_address in local_blocklist:
        return ReputationResult(
            reputation_type=ReputationType.MALICIOUS,
            score=0.0,
            sources=["local_blocklist"],
            indicators=["known_malicious_ip"],
            details={"ip": ip_address},
        )

    # Unknown public IP - neutral
    return ReputationResult(
        reputation_type=ReputationType.UNKNOWN,
        score=0.5,
        sources=["no_match"],
        indicators=["ip_not_in_database"],
        details={"ip": ip_address, "is_private": False},
    )


if __name__ == "__main__":
    # Example: analyze URLs
    test_urls = [
        "https://github.com/project",
        "http://secure-l0gin-verify.xyz/account",
        "https://amaz0n-verify.click/login",
        "http://192.168.1.1/admin",
    ]

    print("URL Reputation Analysis:")
    for url in test_urls:
        result = analyze_url_reputation(url)
        print(f"\n  URL: {url}")
        print(f"  Reputation: {result.reputation_type.value}")
        print(f"  Score: {result.score:.2f}")
        if result.indicators:
            print(f"  Indicators: {', '.join(result.indicators[:3])}")
