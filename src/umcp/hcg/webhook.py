"""
Webhook Orchestrator — Fires rebuild triggers on CONFORMANT events.

When a new weld is executed or a casepack hits the validation ledger as
CONFORMANT, this module fires triggers to rebuild the affected domain sites.

Supported targets:
    - GitHub Actions (repository_dispatch)
    - Vercel deploy hooks
    - Netlify build hooks
    - Generic HTTP POST webhooks

The trigger payload contains the 44 structural identity snapshot + domain
data, giving the static site builder everything it needs for a rebuild.

Usage:
    from umcp.hcg.webhook import WebhookOrchestrator, WebhookTarget

    orch = WebhookOrchestrator()
    orch.add_target(WebhookTarget(
        name="finance-site",
        url="https://api.vercel.com/v1/integrations/deploy/...",
        kind="vercel",
        domains=["finance"],
    ))
    orch.fire("finance", payload={...})
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class WebhookTarget:
    """One deployment target that should be notified on domain changes."""

    name: str
    url: str
    kind: str = "generic"  # generic | github | vercel | netlify
    domains: list[str] = field(default_factory=list)  # empty = all domains
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class WebhookResult:
    """Result of a single webhook fire."""

    target_name: str
    status_code: int
    success: bool
    elapsed_ms: float
    error: str = ""


class WebhookOrchestrator:
    """Manages webhook targets and fires rebuild triggers."""

    def __init__(self) -> None:
        self._targets: list[WebhookTarget] = []

    def add_target(self, target: WebhookTarget) -> None:
        """Register a webhook target."""
        self._targets.append(target)

    def add_target_from_env(self) -> None:
        """Auto-discover targets from environment variables.

        Env format:
            HCG_WEBHOOK_<NAME>_URL=https://...
            HCG_WEBHOOK_<NAME>_KIND=vercel
            HCG_WEBHOOK_<NAME>_DOMAINS=finance,astronomy
            HCG_WEBHOOK_<NAME>_TOKEN=secret (optional, becomes Authorization header)
        """
        prefixes: set[str] = set()
        for key in os.environ:
            if key.startswith("HCG_WEBHOOK_") and key.endswith("_URL"):
                name = key[len("HCG_WEBHOOK_") : -len("_URL")]
                prefixes.add(name)

        for name in sorted(prefixes):
            url = os.environ.get(f"HCG_WEBHOOK_{name}_URL", "")
            if not url:
                continue
            kind = os.environ.get(f"HCG_WEBHOOK_{name}_KIND", "generic")
            domains_str = os.environ.get(f"HCG_WEBHOOK_{name}_DOMAINS", "")
            domains = [d.strip() for d in domains_str.split(",") if d.strip()]
            token = os.environ.get(f"HCG_WEBHOOK_{name}_TOKEN", "")
            headers: dict[str, str] = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            self.add_target(
                WebhookTarget(
                    name=name.lower(),
                    url=url,
                    kind=kind,
                    domains=domains,
                    headers=headers,
                )
            )

    def _build_payload(
        self,
        domain: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the JSON payload for a webhook fire."""
        payload: dict[str, Any] = {
            "event": "domain_rebuild",
            "domain": domain,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "umcp-hcg",
        }
        if extra:
            payload["data"] = extra
        return payload

    def _build_github_payload(
        self,
        domain: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a GitHub repository_dispatch payload."""
        return {
            "event_type": f"hcg-rebuild-{domain}",
            "client_payload": self._build_payload(domain, extra),
        }

    def fire(
        self,
        domain: str,
        payload: dict[str, Any] | None = None,
        *,
        dry_run: bool = False,
    ) -> list[WebhookResult]:
        """Fire webhooks for all targets matching *domain*.

        Parameters
        ----------
        domain : str
            The domain that changed (e.g. "finance").
        payload : dict, optional
            Extra data to include in the webhook body.
        dry_run : bool
            If True, don't actually send requests.

        Returns
        -------
        list[WebhookResult]
            Results for each fired webhook.
        """
        results: list[WebhookResult] = []

        matching = [t for t in self._targets if t.enabled and (not t.domains or domain in t.domains)]

        if not matching:
            logger.info("No webhook targets match domain=%s", domain)
            return results

        for target in matching:
            if target.kind == "github":
                body = self._build_github_payload(domain, payload)
            else:
                body = self._build_payload(domain, payload)

            json_bytes = json.dumps(body).encode("utf-8")

            if dry_run:
                logger.info(
                    "[DRY RUN] Would fire %s → %s (%d bytes)",
                    target.name,
                    target.url,
                    len(json_bytes),
                )
                results.append(
                    WebhookResult(
                        target_name=target.name,
                        status_code=0,
                        success=True,
                        elapsed_ms=0.0,
                    )
                )
                continue

            headers = {
                "Content-Type": "application/json",
                "User-Agent": "UMCP-HCG/2.3.1",
                **target.headers,
            }
            req = Request(
                target.url,
                data=json_bytes,
                headers=headers,
                method="POST",
            )

            t0 = time.monotonic()
            try:
                with urlopen(req, timeout=30) as resp:
                    status = resp.status
                elapsed = (time.monotonic() - t0) * 1000
                results.append(
                    WebhookResult(
                        target_name=target.name,
                        status_code=status,
                        success=200 <= status < 300,
                        elapsed_ms=elapsed,
                    )
                )
                logger.info(
                    "Webhook %s → %d (%.1f ms)",
                    target.name,
                    status,
                    elapsed,
                )
            except (URLError, OSError) as exc:
                elapsed = (time.monotonic() - t0) * 1000
                results.append(
                    WebhookResult(
                        target_name=target.name,
                        status_code=0,
                        success=False,
                        elapsed_ms=elapsed,
                        error=str(exc),
                    )
                )
                logger.warning(
                    "Webhook %s failed: %s (%.1f ms)",
                    target.name,
                    exc,
                    elapsed,
                )

        return results

    @property
    def targets(self) -> list[WebhookTarget]:
        """Return registered targets (read-only copy)."""
        return list(self._targets)
