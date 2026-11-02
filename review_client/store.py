"""Local, versioned crossover review. Original labels are never modified."""
import csv
import hashlib
import io
import json
import random
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CRITERIA = {
    "version": "v3-2026-11-02-depth-background-allowed",
    "rules": ["Различимы и мелкие, и крупные сосуды.",
              "Границы сосудов различимы, а не только их общий ход.",
              "Основная сеть чёткая по всему полю. Размытые глубокие фоновые сосуды допустимы.",
              "Частично резкая основная сеть не подходит: её границы должны быть различимы по всему полю.",
              "«Не уверен» — неуверенность оценивающего; такой ответ не становится обучающим классом."],
}
LABELS = ("ok", "not_ok", "uncertain")


def now():
    return datetime.now(timezone.utc).isoformat()


def old_label(value):
    if value is True:
        return "ok"
    if value is False:
        return "not_ok"
    if value == "unknown":
        return "uncertain"
    raise ValueError("Unexpected source label")


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, dataset, state):
        self.dataset, self.state = Path(dataset).resolve(), Path(state).resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        raw = (self.dataset / "frames.json").read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        snapshot = self.state / "source_snapshot.json"
        if snapshot.exists():
            saved = json.loads(snapshot.read_text(encoding="utf-8"))
            if saved["annotation_sha256"] != self.digest or saved["criteria"] != CRITERIA:
                raise ValueError("Source labels or criteria changed. Use a new review state directory.")
        else:
            frames = json.loads(raw)
            with (self.dataset / "patient_map.csv").open(encoding="utf-8-sig", newline="") as f:
                patients = {r["video"]: r["patient_id"] for r in csv.DictReader(f)}
            records = []
            for frame in frames:
                file = frame["file"]
                if Path(file).name != file or "/" in file or "\\" in file:
                    raise ValueError("Image filename must be a basename")
                if frame["annotator"] not in ("human", "assistant"):
                    raise ValueError("Unknown original annotator")
                if not (self.dataset / "pictures" / file).is_file():
                    raise FileNotFoundError(file)
                video = Path(file).stem.rsplit("_", 1)[0] + ".avi"
                records.append(dict(id=hashlib.sha256(file.encode()).hexdigest()[:20], file=file,
                                    video=video, patient_id=patients[video], original_source=frame["annotator"],
                                    original_label=old_label(frame["quality_ok"]),
                                    assigned_to="human" if frame["annotator"] == "assistant" else "assistant"))
            if len({r["id"] for r in records}) != len(records):
                raise ValueError("Duplicate records")
            random.Random(20261102).shuffle(records)
            saved = dict(annotation_sha256=self.digest, criteria=CRITERIA, created_at=now(), records=records)
            snapshot.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        self.records = saved["records"]
        self.by_id = {r["id"]: r for r in self.records}
        self.db = self.state / "reviews.sqlite3"
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS reviews (
                  image_id TEXT PRIMARY KEY, reviewer TEXT NOT NULL,
                  label TEXT NOT NULL CHECK(label IN ('ok','not_ok','uncertain')),
                  reasons TEXT NOT NULL, comment TEXT NOT NULL,
                  updated_at TEXT NOT NULL, revision INTEGER NOT NULL,
                  criteria_version TEXT NOT NULL, method TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT, image_id TEXT NOT NULL,
                  reviewer TEXT NOT NULL, previous_json TEXT, next_json TEXT NOT NULL,
                  created_at TEXT NOT NULL);
            """)

    def connect(self):
        con = sqlite3.connect(self.db, timeout=15)
        con.row_factory = sqlite3.Row
        return con

    def all_reviews(self):
        with self.connect() as con:
            return {r["image_id"]: dict(r) for r in con.execute("SELECT * FROM reviews")}

    def status(self):
        reviews = self.all_reviews()
        counts = {}
        for role in ("human", "assistant"):
            records = [r for r in self.records if r["assigned_to"] == role]
            counts[role] = dict(total=len(records), done=sum(r["id"] in reviews for r in records))
        counts["complete"] = all(counts[r]["done"] == counts[r]["total"] for r in ("human", "assistant"))
        return counts

    def human_state(self):
        reviews = self.all_reviews()
        items = []
        for r in self.records:
            if r["assigned_to"] != "human":
                continue
            review = reviews.get(r["id"])
            items.append(dict(id=r["id"], label=review["label"] if review else None,
                              reasons=json.loads(review["reasons"]) if review else [],
                              comment=review["comment"] if review else "",
                              revision=review["revision"] if review else 0))
        # Neither old labels nor filenames nor other reviewer's decisions are exposed.
        return dict(items=items, criteria=CRITERIA, progress=self.status())

    def save(self, image_id, reviewer, label, reasons=(), comment="", revision=0, method="visual_native_client"):
        if image_id not in self.by_id or self.by_id[image_id]["assigned_to"] != reviewer:
            raise ValueError("Image is not assigned to this reviewer")
        if label not in LABELS or not isinstance(comment, str) or len(comment) > 2000:
            raise ValueError("Invalid label or comment")
        if not isinstance(reasons, (list, tuple)) or len(reasons) > 10 or any(not isinstance(s, str) or len(s) > 120 for s in reasons):
            raise ValueError("Invalid reasons")
        if not isinstance(revision, int) or revision < 0:
            raise ValueError("Invalid revision")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT * FROM reviews WHERE image_id=?", (image_id,)).fetchone()
            old = dict(old) if old else None
            if revision != (old["revision"] if old else 0):
                raise Conflict("Эта оценка уже изменена в другой вкладке. Обновите страницу.")
            row = dict(image_id=image_id, reviewer=reviewer, label=label,
                       reasons=json.dumps(list(reasons), ensure_ascii=False), comment=comment,
                       updated_at=now(), revision=revision + 1, criteria_version=CRITERIA["version"], method=method)
            con.execute("INSERT OR REPLACE INTO reviews VALUES (:image_id,:reviewer,:label,:reasons,:comment,:updated_at,:revision,:criteria_version,:method)", row)
            con.execute("INSERT INTO events(image_id,reviewer,previous_json,next_json,created_at) VALUES(?,?,?,?,?)",
                        (image_id, reviewer, json.dumps(old, ensure_ascii=False) if old else None,
                         json.dumps(row, ensure_ascii=False), now()))
        return row

    def comparison(self):
        reviews = self.all_reviews()
        rows = []
        for record in self.records:
            new = reviews.get(record["id"])
            if not new:
                status = "pending"
            elif new["label"] != record["original_label"]:
                status = "disagreement"
            elif new["label"] == "uncertain":
                status = "uncertain_agreement"
            else:
                status = "agreed_binary"
            human = record["original_label"] if record["original_source"] == "human" else new["label"] if new else None
            assistant = record["original_label"] if record["original_source"] == "assistant" else new["label"] if new else None
            rows.append(dict(**record, human_label=human, assistant_label=assistant,
                             new_label=new["label"] if new else None,
                             review_reason="; ".join(json.loads(new["reasons"])) if new else "",
                             review_comment=new["comment"] if new else "", status=status))
        return rows

    def summary(self):
        rows = self.comparison()
        counts = {s: sum(r["status"] == s for r in rows) for s in ("pending", "agreed_binary", "uncertain_agreement", "disagreement")}
        counts["agreed_ok"] = sum(r["status"] == "agreed_binary" and r["human_label"] == "ok" for r in rows)
        counts["agreed_not_ok"] = sum(r["status"] == "agreed_binary" and r["human_label"] == "not_ok" for r in rows)
        matrix = [[sum(r["human_label"] == h and r["assistant_label"] == a for r in rows) for a in LABELS] for h in LABELS]
        n = sum(map(sum, matrix))
        observed = sum(matrix[i][i] for i in range(3)) / n if n else None
        expected = sum(sum(matrix[i]) * sum(row[i] for row in matrix) for i in range(3)) / (n * n) if n else None
        kappa = (observed - expected) / (1 - expected) if expected is not None and expected < 1 else None
        return dict(counts=counts, class_order=LABELS, matrix_human_rows_assistant_columns=matrix,
                    agreement_rate=observed, kappa=kappa, progress=self.status(),
                    note="Перекрёстная проверка исторических меток по уточнённым критериям. Совпадение не доказывает истинность метки; отбор исключает трудные случаи.")

    @staticmethod
    def csv_bytes(rows):
        f = io.StringIO(newline="")
        fields = list(rows[0]) if rows else ["file", "quality_ok", "patient_id"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return ("\ufeff" + f.getvalue()).encode("utf-8")

    def export(self, include_images=False):
        if not self.status()["complete"]:
            raise ValueError("Экспорт совпадений доступен после завершения обеих очередей.")
        rows = self.comparison()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = self.state / "exports" / ("consensus_" + stamp)
        folder.mkdir(parents=True)
        agreed = [r for r in rows if r["status"] == "agreed_binary"]
        frames = [dict(file=r["file"], quality_ok=r["human_label"] == "ok", patient_id=r["patient_id"], video=r["video"],
                       human_label=r["human_label"], assistant_label=r["assistant_label"],
                       original_source=r["original_source"], review_criteria=CRITERIA["version"]) for r in agreed]
        (folder / "frames.json").write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "consensus.csv").write_bytes(self.csv_bytes(frames))
        (folder / "all_comparisons.csv").write_bytes(self.csv_bytes(rows))
        for status in ("disagreement", "uncertain_agreement"):
            (folder / (status + ".json")).write_text(json.dumps([r for r in rows if r["status"] == status], ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "criteria.json").write_text(json.dumps(CRITERIA, ensure_ascii=False, indent=2), encoding="utf-8")
        video_map = {r["video"]: r["patient_id"] for r in agreed}
        (folder / "patient_map.csv").write_bytes(self.csv_bytes([dict(video=v, patient_id=p) for v, p in video_map.items()]))
        if include_images:
            pictures = folder / "pictures"
            pictures.mkdir()
            for r in agreed:
                shutil.copy2(self.dataset / "pictures" / r["file"], pictures / r["file"])
        else:
            (folder / "IMAGE_ROOT.txt").write_text(str(self.dataset / "pictures"), encoding="utf-8")
        (folder / "README.txt").write_text(
            "Согласованные определённые ответы: ok и not_ok. Неуверенность и расхождения сохранены отдельно.\n"
            "Один ответ на кадр — исторический, второй — новая перекрёстная оценка.\n"
            "Отбор по согласию смещает набор в сторону простых случаев. Не используйте только этот набор как финальный тест.\n"
            "Все кадры одного пациента сохраняйте в одной части train/validation/test.\n", encoding="utf-8")
        return dict(path=str(folder), count=len(frames), images_copied=include_images)
