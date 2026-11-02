"""One-time, audited correction requested by the human annotator.

Run with the annotation server stopped. Only new human answers are swapped.
"""
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

MIGRATION = "human_keys_2_uncertain_3_not_ok_20261102"


def correct(state):
    state = Path(state)
    database = state / "reviews.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    con = sqlite3.connect(database, timeout=15)
    con.row_factory = sqlite3.Row
    try:
        exists = con.execute("SELECT name FROM sqlite_master WHERE name='maintenance'").fetchone()
        if exists:
            previous = con.execute("SELECT report FROM maintenance WHERE name=?", (MIGRATION,)).fetchone()
            if previous:
                return dict(json.loads(previous[0]), already_applied=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = state / "backups" / ("before_key_correction_" + stamp)
        backup.mkdir(parents=True)
        target = sqlite3.connect(backup / "reviews.sqlite3")
        try:
            con.backup(target)
        finally:
            target.close()
        shutil.copy2(state / "source_snapshot.json", backup / "source_snapshot.json")
        timestamp = datetime.now(timezone.utc).isoformat()
        with con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("CREATE TABLE IF NOT EXISTS maintenance(name TEXT PRIMARY KEY, report TEXT NOT NULL)")
            before = {r["image_id"]: dict(r) for r in con.execute("SELECT * FROM reviews")}
            counts = {"not_ok_to_uncertain": 0, "uncertain_to_not_ok": 0, "ok_unchanged": 0}
            mapping = {"not_ok": "uncertain", "uncertain": "not_ok"}
            for old in before.values():
                if old["reviewer"] != "human":
                    continue
                if old["label"] == "ok":
                    counts["ok_unchanged"] += 1
                    continue
                new = dict(old, label=mapping[old["label"]], revision=old["revision"] + 1,
                           updated_at=timestamp, method=MIGRATION)
                counts[old["label"] + "_to_" + new["label"]] += 1
                con.execute("UPDATE reviews SET label=:label, revision=:revision, updated_at=:updated_at, method=:method WHERE image_id=:image_id", new)
                con.execute("INSERT INTO events(image_id,reviewer,previous_json,next_json,created_at) VALUES(?,?,?,?,?)",
                            (old["image_id"], "human", json.dumps(old, ensure_ascii=False),
                             json.dumps(new, ensure_ascii=False), timestamp))
            after = {r["image_id"]: dict(r) for r in con.execute("SELECT * FROM reviews")}
            assert before.keys() == after.keys()
            for key, old in before.items():
                if old["reviewer"] != "human" or old["label"] == "ok":
                    assert after[key] == old
                else:
                    assert after[key]["label"] == mapping[old["label"]]
                    assert after[key]["comment"] == old["comment"]
                    assert after[key]["reasons"] == old["reasons"]
            report = dict(migration=MIGRATION, applied_at=timestamp, backup=str(backup.resolve()),
                          counts=counts, total_human=sum(r["reviewer"] == "human" for r in before.values()),
                          assistant_unchanged=sum(r["reviewer"] == "assistant" for r in before.values()))
            con.execute("INSERT INTO maintenance VALUES(?,?)", (MIGRATION, json.dumps(report, ensure_ascii=False)))
        (backup / "correction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        con.close()


if __name__ == "__main__":
    print(json.dumps(correct(Path(__file__).resolve().parent / "data"), ensure_ascii=True))
