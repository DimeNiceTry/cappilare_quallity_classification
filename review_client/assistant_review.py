"""Blinded assistant queue and explicit visually reviewed annotation import."""
import argparse
import json
from pathlib import Path
from store import Store

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("queue", "import", "status"))
    parser.add_argument("--file", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    store = Store(ROOT.parent / "capillary_quality_1000", ROOT / "data")
    queue = [r for r in store.records if r["assigned_to"] == "assistant"]
    if args.command == "queue":
        reviews = store.all_reviews()
        rows = [dict(index=i, id=r["id"], path=str(store.dataset / "pictures" / r["file"]), reviewed=r["id"] in reviews)
                for i, r in enumerate(queue)]
        print(json.dumps(rows[args.start:args.start + args.count], ensure_ascii=True))
    elif args.command == "import":
        if args.file is None:
            parser.error("--file is required")
        rows = json.loads(args.file.read_text(encoding="utf-8"))
        for row in rows:
            r = queue[row["index"]]
            if row.get("id", r["id"]) != r["id"]:
                raise ValueError("Queue ID mismatch")
            store.save(r["id"], "assistant", row["label"], row.get("reasons", []), row.get("comment", ""),
                       row.get("revision", 0), method="assistant_visual_review_original_png")
        print(json.dumps(store.status()))
    else:
        print(json.dumps(store.status()))


if __name__ == "__main__":
    main()
