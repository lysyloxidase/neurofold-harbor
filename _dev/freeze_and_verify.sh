#!/bin/sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASKS="alzheimer-abeta42-v8 huntington-htt-polyq-v8 parkinson-alpha-synuclein-v8 alzheimer-tau-v8"
for T in $TASKS; do
  echo ""; echo "########## FREEZE $T ##########"; date
  python3 _dev/freeze_task.py --task "$T" --episodes 128 --workers 5 2>&1 | tail -5
done
for T in $TASKS; do
  echo ""; echo "########## DOCKER ORACLE $T ##########"
  docker build -q -t nf8-$T-env  ./$T/environment >/dev/null 2>&1 || { echo "ENV BUILD FAIL"; continue; }
  docker build -q -t nf8-$T-tests ./$T/tests      >/dev/null 2>&1 || { echo "TESTS BUILD FAIL"; continue; }
  docker volume rm -f nf8v >/dev/null 2>&1; docker volume create nf8v >/dev/null
  docker run --rm --network none -v nf8v:/logs -v "$PWD/$T/solution":/solution:ro \
    nf8-$T-tests sh /solution/solve.sh >/dev/null 2>&1
  docker run --rm --network none -v nf8v:/logs nf8-$T-tests bash /tests/test.sh >/dev/null 2>&1
  R=$(docker run --rm -v nf8v:/logs nf8-$T-tests cat /logs/verifier/reward.txt 2>/dev/null)
  E=$(docker run --rm -v nf8v:/logs nf8-$T-tests python -c "import json;d=json.load(open('/logs/verifier/metrics.json'));print(round(d.get('extended_score',-9),6), d.get('error','')[:40])" 2>/dev/null)
  echo "  reward=${R:-BRAK}   extended/err: $E"
done
echo ""; echo "########## ZAKONCZONE ##########"; date
