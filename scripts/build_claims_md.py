"""Render claims/claims.json as docs/CLAIMS.md (the human-readable claim ledger)."""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
c = json.loads((ROOT / "claims/claims.json").read_text())
label = {"established": "Established", "documented_limitation": "Documented limitation", "under_validation": "Under validation", "not_established": "Not established"}
out = ["# Claim ledger", "", "Generated from `claims/claims.json` by `scripts/build_claims_md.py`; do not edit by hand.", "",
       c["note"], "", "| ID | Claim | Evidence | Test | Status |", "|---|---|---|---|---|"]
for x in c["claims"]:
    out.append(f"| {x['id']} | {x['statement']} | {x['evidence'] or '—'} | {('`' + x['test'] + '`') if x['test'] else '—'} | "
               f"{label[x['status']]} |")
(ROOT / "docs/CLAIMS.md").write_text("\n".join(out) + "\n")
print("docs/CLAIMS.md written")
