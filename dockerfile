# -------- Stage 1: build & fetch a single model --------
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Install curl + certs and Ollama (no recommends, clean lists)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

# Model store + choose ONE model
ENV OLLAMA_MODELS=/models
ENV OLLAMA_MODEL=mistral:7b
RUN mkdir -p "$OLLAMA_MODELS"

# Pull model; clean caches in the SAME layer
RUN ( \
  OLLAMA_MODELS="$OLLAMA_MODELS" nohup ollama serve >/tmp/ollama.log 2>&1 & \
  pid=$!; sleep 3; \
  OLLAMA_MODELS="$OLLAMA_MODELS" ollama pull "$OLLAMA_MODEL"; \
  kill $pid; wait $pid || true \
) && rm -rf /root/.ollama/cache /root/.ollama/tmp /tmp/*

# If /usr/local/lib/ollama is heavy, prune unneeded dirs (adjust after inspecting):
# RUN rm -rf /usr/local/lib/ollama/{cache,tmp,tests,examples,docs,models,blobs} || true

# -------- Stage 2: runtime (offline) --------
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Only certs (and tini if you want a proper init)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy only what you need
COPY --from=builder /usr/local/bin/ollama /usr/local/bin/ollama
COPY --from=builder /usr/local/lib/ollama /usr/local/lib/ollama

# Create non-root first, then COPY models with ownership set to avoid copy-up
RUN useradd -m -u 10001 app
COPY --from=builder --chown=10001:10001 /models /models

USER app
ENV OLLAMA_MODELS=/models
ENV OLLAMA_HOST=0.0.0.0:11434

# Simple healthcheck: list local models
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD /usr/local/bin/ollama list || exit 1

EXPOSE 11434
ENTRYPOINT ["ollama", "serve"]