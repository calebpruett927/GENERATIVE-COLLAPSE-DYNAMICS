# UMCP Development Container — v2.1.3
# Python 3.12 on Debian Bookworm — matches CI (ubuntu-latest / Python 3.12).
# Includes: LaTeX (RevTeX4-2 papers), system libs for scipy/numpy, git tooling.

FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# ── Remove stale third-party repos from base image ──────────────
# The base image may ship Yarn/Node apt repos with expired GPG keys.
# Not needed for this project — remove before any apt operations.
RUN rm -f /etc/apt/sources.list.d/yarn* \
         /etc/apt/sources.list.d/*yarn* \
         /etc/apt/keyrings/yarn* \
         /etc/apt/trusted.gpg.d/yarn* 2>/dev/null; true

# ── System packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # LaTeX — sufficient for RevTeX4-2 papers + BibTeX
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-publishers \
    texlive-science \
    texlive-bibtex-extra \
    biber \
    latexmk \
    chktex \
    # Build tools (scipy/numpy compilation fallback)
    build-essential \
    gfortran \
    # CLI utilities
    jq \
    tree \
    bat \
    fd-find \
    ripgrep \
    shellcheck \
    # Networking (API/dashboard debugging)
    curl \
    httpie \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Convenience symlinks ─────────────────────────────────────────
# Debian packages bat/fd under non-standard names
RUN ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd

# ── Python tooling (system-level, always available) ──────────────
RUN pip install --no-cache-dir \
    "pip>=24.0" \
    "setuptools>=68" \
    wheel

# ── Pre-install heavy wheels to speed up postCreateCommand ───────
RUN pip install --no-cache-dir \
    "numpy>=1.24.0" \
    "scipy>=1.10.0"

# ── Workspace ────────────────────────────────────────────────────
WORKDIR /workspaces/UMCP-Metadata-Runnable-Code

LABEL maintainer="Clement Paulus"
LABEL description="UMCP development container — Python 3.12 + LaTeX + full toolchain"
LABEL version="2.1.3"
