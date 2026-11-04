# One image, ten services: the service is chosen by the container command (python -m p2p.serve <service>).
#   open stack : docker build --build-arg EXTRAS=ollama -t p2p .
#   GCP        : docker build --build-arg EXTRAS=gcp    -t REGION-docker.pkg.dev/PROJECT/p2p/app:TAG .
FROM python:3.12-slim AS build
ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install --no-cache ".${EXTRAS:+[$EXTRAS]}"

FROM python:3.12-slim
# libgomp1: LightGBM runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 10001 app && mkdir -p /data && chown app /data
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/data
USER app
WORKDIR /home/app
# Cloud Run injects $PORT; locally each service falls back to its default port (see p2p/serve.py).
HEALTHCHECK --interval=15s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % (os.environ.get('PORT') or os.environ.get('HEALTH_PORT','8100')), timeout=2)"
ENTRYPOINT ["python", "-m", "p2p.serve"]
CMD ["gateway"]
