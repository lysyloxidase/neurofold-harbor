FROM python:3.12.11-slim-bookworm
ENV PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
WORKDIR /tests
COPY . /tests
RUN pip install --no-cache-dir numpy==2.2.6 && mkdir -p /logs/verifier
CMD ["sleep", "infinity"]
