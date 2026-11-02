"""Record the user's explicit clarification without erasing earlier provenance."""
import json
import shutil
from pathlib import Path
from store import CRITERIA, now


def migrate(state):
    path = Path(state) / "source_snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot["criteria"] == CRITERIA:
        return
    if snapshot["criteria"]["version"] != "v2-2026-11-02":
        raise ValueError("Unexpected criteria; refusing implicit migration")
    archive = Path(state) / "criteria_history"
    archive.mkdir(exist_ok=True)
    old = archive / "source_snapshot_v2.json"
    if not old.exists():
        shutil.copy2(path, old)
    snapshot.setdefault("criteria_history", []).append(dict(
        criteria=snapshot["criteria"], clarified_at=now(),
        reason="User explicitly allowed blurred deep background when the main vessel network is clear. Earlier review rows retain their original criteria version."))
    snapshot["criteria"] = CRITERIA
    temp = path.with_suffix(".clarified.tmp")
    temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


if __name__ == "__main__":
    migrate(Path(__file__).resolve().parent / "data")
    print("Depth-background clarification recorded")
