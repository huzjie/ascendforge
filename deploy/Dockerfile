# ascendforge runtime image
FROM python:3.11-slim

WORKDIR /app

COPY ascendforge/ ./ascendforge/
COPY examples/ ./examples/
COPY configs/ ./configs/
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir . || true

EXPOSE 8000

ENV ASCENDFORGE_BACKEND=mock

CMD ["python", "-m", "ascendforge", "serve"]
