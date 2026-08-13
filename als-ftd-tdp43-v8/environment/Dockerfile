FROM python:3.12.11-slim-bookworm
ENV PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir numpy==2.2.6 \
 && python -c "import policy_runtime as p; p.save_policy(p.zero_policy(), '/app/policy.json')" \
 && mkdir -p /logs/artifacts
CMD ["sleep", "infinity"]
