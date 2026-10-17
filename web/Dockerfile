# Public site: play against the bot. Build from the repository root:
#   docker build -f web/Dockerfile -t catan-bots . && docker run -p 8000:8000 catan-bots
# Only the public files are copied in: the advisor (web/private/) is not in the image.
# Games against people are recorded in /app/games/human; keep them across restarts with a volume:
#   docker run -p 8000:8000 -v catan-games:/app/games catan-bots
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "catanatron @ git+https://github.com/bcollazo/catanatron.git@d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816" \
    networkx==3.6.1 numpy==2.4.6
WORKDIR /app
COPY harness/__init__.py harness/_patch.py harness/obs.py harness/host.py harness/
COPY tools/advise.py tools/
COPY agents/distill/ agents/distill/
COPY web/server.py web/
COPY web/static/ web/static/
RUN useradd -m app && mkdir -p /app/games/human && chown -R app /app/games
USER app
EXPOSE 8000
CMD ["python", "web/server.py", "--port", "8000"]
