#!/bin/sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step(){ echo ""; echo "########## $* ##########"; date; }

step "bramka TDP-43"
python3 _dev/porting_gate.py --task als-ftd-tdp43-v8 --workers 5 2>&1 | tail -16
step "bramka tau"
python3 _dev/porting_gate.py --task alzheimer-tau-v8 --workers 5 2>&1 | tail -16

for T in huntington-htt-polyq-v8 parkinson-alpha-synuclein-v8 als-ftd-tdp43-v8 alzheimer-tau-v8; do
  BLOCK=$(python3 -c "
import json;d=json.load(open('agentic/reports/validation/porting_gate_$T.json'))
print(','.join(d.get('blocking',[])) or 'none')" 2>/dev/null || echo unknown)
  if [ "$BLOCK" = "none" ]; then
    step "referencja $T (bramka nie blokuje)"
    python3 _dev/train_reference.py --task "$T" --budget 12000 --restarts 2 --workers 5 2>&1 | tail -4
  else
    step "POMINIETO referencje $T — bramka blokuje: $BLOCK"
  fi
done
echo ""; echo "########## LANCUCH ZAKONCZONY ##########"; date
