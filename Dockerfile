# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /PROJECT

# ── Dependencies ───────────────────────────────────────────────────────────────
# Hint: copy requirements.txt BEFORE copying the rest of the code.
# Why? Docker caches each layer. If you copy code first, any code change
# invalidates the pip install cache and reinstalls everything from scratch.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ── Application files ──────────────────────────────────────────────────────────
COPY . .


# ── Port ───────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Startup command ────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
