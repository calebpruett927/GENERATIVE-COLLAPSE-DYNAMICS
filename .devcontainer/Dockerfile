# UMCP Development Container
# Python 3.12 on Debian Bookworm — matches CI (ubuntu-latest / Python 3.12).
# Includes: LaTeX (RevTeX4-2 papers), system libs for scipy/numpy, git tooling.

FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────────
# - texlive: LaTeX compilation for paper/ (RevTeX4-2, BibTeX)
# - build-essential, gfortran: scipy/numpy compilation fallback
# - jq: JSON processing for casepack manifests and validator reports
# - tree: directory visualization
# - bat: syntax-highlighted file viewing
# - fd-find: fast file search (aliased to fd)
# - ripgrep: fast grep alternative
# - shellcheck: shell script linting for scripts/*.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
    # LaTeX — minimal but sufficient for RevTeX4-2 papers
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-publishers \
    texlive-science \
    texlive-bibtex-extra \
    biber \
    latexmk \
    # Build tools for compiled Python deps (numpy, scipy wheels)
    build-essential \
    gfortran \
    # CLI utilities
    jq \
    tree \
    bat \
    fd-find \
    ripgrep \
    shellcheck \
    # Networking (for API/dashboard debugging)
    curl \
    httpie \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Convenience symlinks ─────────────────────────────────────────
# Debian packages use non-standard binary names
RUN ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd

# ── Python tooling (system-level, always available) ──────────────
# Pinned to match CI exactly. Available before project install.
RUN pip install --no-cache-dir \
    pip>=24.0 \
    setuptools>=68 \
    wheel

# ── Pre-install heavy wheels to speed up postCreateCommand ───────
# numpy/scipy take longest; pre-caching them cuts rebuild time ~60s.
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0

# ── Workspace configuration ─────────────────────────────────────
WORKDIR /workspaces/UMCP-Metadata-Runnable-Code

# Labels for image metadata
LABEL maintainer="Clement Paulus"
LABEL description="UMCP development container — Python 3.12 + LaTeX + full toolchain"
LABEL version="2.0.0"
