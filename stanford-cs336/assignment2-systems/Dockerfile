# Dockerfile
FROM vastai/pytorch
WORKDIR /app

# Install uv (pick ONE of these; try pip first)
RUN python -m pip install --upgrade pip && pip install uv

# Copy only dependency manifests first for layer caching
# Adjust these filenames to what your repo actually uses
COPY pyproject.toml uv.lock* /app/

# Create the venv INSIDE the image
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install deps (cache-friendly)
RUN uv sync --frozen || uv sync

# Now copy the rest of your repo (including .git if not ignored)
COPY . /app

CMD ["bash"]