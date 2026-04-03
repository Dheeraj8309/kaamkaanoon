# Dockerfile
# This file is the recipe for building our Docker container.
# It tells Docker exactly how to set up the environment to run KaamKaanoon.
# Build command: docker build -t kaamkaanoon .
# Run command: docker run -p 8501:8501 --env-file .env kaamkaanoon

# ── Step 1: Choose the base image ──
# We start from an official Python 3.12 image
# "slim" means a minimal version — smaller size, faster build
FROM python:3.12-slim

# ── Step 2: Set the working directory ──
# All commands from here on run inside /app inside the container
# Think of this as our project root inside Docker
WORKDIR /app

# ── Step 3: Install system dependencies ──
# These are Linux packages needed by some of our Python libraries
# chromadb and sentence-transformers need these to compile
RUN apt-get update && apt-get install -y \
    # gcc is needed to compile some Python packages
    gcc \
    # g++ is needed for C++ dependencies
    g++ \
    # clean up apt cache to keep image size small
    && rm -rf /var/lib/apt/lists/*

# ── Step 4: Copy requirements first ──
# We copy requirements.txt BEFORE copying the rest of the code
# This is a Docker best practice — if requirements don't change,
# Docker reuses the cached layer and skips reinstalling everything
COPY requirements.txt .

# ── Step 5: Install Python dependencies ──
# --no-cache-dir keeps the image smaller by not storing pip cache
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 6: Copy the entire project into the container ──
# The . on the left means "everything in current folder on your machine"
# The . on the right means "copy it to /app inside the container"
COPY . .

# ── Step 7: Set environment variables inside the container ──
# These suppress the warnings we've been seeing throughout the project
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV ANONYMIZED_TELEMETRY=False

# ── Step 8: Expose the port Streamlit runs on ──
# Streamlit uses port 8501 by default
# This tells Docker to make that port accessible from outside the container
EXPOSE 8501

# ── Step 9: Health check ──
# Docker periodically checks if the app is still running
# This pings the Streamlit health endpoint every 30 seconds
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Step 10: The command that starts our app ──
# This runs when someone does "docker run kaamkaanoon"
# --server.address=0.0.0.0 makes it accessible from outside the container
# --server.port=8501 sets the port explicitly
# --server.headless=true disables browser auto-open inside container
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]