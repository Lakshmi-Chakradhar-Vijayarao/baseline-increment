#!/usr/bin/env bash
# Build a self-contained Kaggle kernel (score.py + kaggle_kernel.py) and optionally push it.
#   KAGGLE_USER=<name> ./build_kernel.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
: "${KAGGLE_USER:?set KAGGLE_USER}"
out=../../dist/kaggle_ragtruth && rm -rf "$out" && mkdir -p "$out"
{ sed '/^if __name__ == "__main__":/,$d' score.py; echo; echo "# ---- runner ----"
  sed -e '/score_src = next/,/score_src = here \/ "score.py"/d' \
      -e 's|subprocess.run(\[sys.executable, str(score_src), str(d), str(here / "ragtruth_lettucedetect.csv")\], check=True)|main(str(d), str(here / "ragtruth_lettucedetect.csv"))|' \
      -e '/^SCORE_PY = /d' kaggle_kernel.py | sed '1,/^"""$/{/^"""/,/^"""$/d;}'; } > "$out/kernel.py"
sed -e "s/<KAGGLE_USER>/$KAGGLE_USER/" -e 's/kaggle_kernel.py/kernel.py/' kernel-metadata.example.json > "$out/kernel-metadata.json"
python3 -c "import ast,sys; ast.parse(open('$out/kernel.py').read())" && echo "built $out/kernel.py"
if [[ "${1:-}" == "--push" ]]; then kaggle kernels push -p "$out" --accelerator NvidiaTeslaT4; fi
