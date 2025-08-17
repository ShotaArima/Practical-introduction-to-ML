# uv の公式イメージをベースにする
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# 必要ツールの導入（ビルドに必要な依存）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl build-essential pkg-config python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Rust toolchain（非対話）を導入
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# PATH を通す（ログインシェルでなくても使えるように）
ENV PATH="/root/.cargo/bin:${PATH}"

# 以降、必要なら依存を固定して同期
# ※ 既に pyproject.toml / uv.lock があるなら、ここで同期してキャッシュを効かせるのが高速
# WORKDIR /app
# COPY pyproject.toml uv.lock ./
# RUN uv sync --frozen
# COPY . .
