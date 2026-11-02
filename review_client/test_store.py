import csv
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from store import Conflict, Store
from server import make_server
from correct_key_mixup import correct


class ReviewChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.data, self.state = root / "dataset", root / "state"
        (self.data / "pictures").mkdir(parents=True)
        frames = []
        for i, label in enumerate([True, False, "unknown", True, False, "unknown"]):
            file = f"video{i}_1.png"
            (self.data / "pictures" / file).write_bytes(b"original image bytes")
            frames.append(dict(file=file, quality_ok=label, annotator="human" if i < 3 else "assistant"))
        (self.data / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
        with (self.data / "patient_map.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video", "patient_id"])
            w.writeheader()
            w.writerows(dict(video=f"video{i}.avi", patient_id=str(i)) for i in range(6))
        self.store = Store(self.data, self.state)

    def tearDown(self):
        self.temp.cleanup()

    def test_assignment_and_hidden_labels(self):
        state = self.store.human_state()
        self.assertEqual(len(state["items"]), 3)
        for row in state["items"]:
            self.assertEqual(self.store.by_id[row["id"]]["original_source"], "assistant")
            self.assertIsNone(row["label"])
            self.assertNotIn("file", row)
            self.assertNotIn("original_label", row)
        forbidden = next(r for r in self.store.records if r["assigned_to"] == "assistant")
        with self.assertRaises(ValueError):
            self.store.save(forbidden["id"], "human", "ok")

    def test_durable_edits_and_stale_tab_rejection(self):
        r = self.store.records[0]
        self.store.save(r["id"], r["assigned_to"], "not_ok", ["blur"], "first")
        restored = Store(self.data, self.state)
        self.assertEqual(restored.all_reviews()[r["id"]]["comment"], "first")
        with self.assertRaises(Conflict):
            restored.save(r["id"], r["assigned_to"], "ok", revision=0)
        restored.save(r["id"], r["assigned_to"], "uncertain", revision=1)
        with restored.connect() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)
        self.assertEqual(restored.status()[r["assigned_to"]]["done"], 1)

    def test_human_key_correction_is_scoped_backed_up_and_idempotent(self):
        for r in self.store.records:
            self.store.save(r["id"], r["assigned_to"], r["original_label"], ["reason"], "comment")
        before = self.store.all_reviews()
        report = correct(self.state)
        self.assertEqual(report["total_human"], 3)
        self.assertEqual(report["counts"], {"not_ok_to_uncertain": 1, "uncertain_to_not_ok": 1, "ok_unchanged": 1})
        self.assertTrue((Path(report["backup"]) / "reviews.sqlite3").is_file())
        after = self.store.all_reviews()
        for key, old in before.items():
            new = after[key]
            if old["reviewer"] == "assistant" or old["label"] == "ok":
                self.assertEqual(old, new)
            else:
                self.assertNotEqual(old["label"], new["label"])
                self.assertEqual(old["comment"], new["comment"])
                self.assertEqual(old["reasons"], new["reasons"])
                self.assertEqual(old["revision"] + 1, new["revision"])
        self.assertTrue(correct(self.state)["already_applied"])
        self.assertEqual(after, self.store.all_reviews())

    def test_export_excludes_uncertainty_and_disagreement(self):
        original = (self.data / "frames.json").read_bytes()
        with self.assertRaises(ValueError):
            self.store.export()
        for r in self.store.records:
            label = "not_ok" if r["file"] == "video3_1.png" else r["original_label"]
            self.store.save(r["id"], r["assigned_to"], label)
        summary = self.store.summary()
        self.assertEqual(summary["counts"]["agreed_binary"], 3)
        self.assertEqual(summary["counts"]["uncertain_agreement"], 2)
        self.assertEqual(summary["counts"]["disagreement"], 1)
        exported = Path(self.store.export(True)["path"])
        frames = json.loads((exported / "frames.json").read_text())
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(type(r["quality_ok"]) is bool for r in frames))
        self.assertEqual(len(list((exported / "pictures").glob("*.png"))), 3)
        self.assertEqual((self.data / "frames.json").read_bytes(), original)

    def test_source_changes_do_not_silently_reuse_reviews(self):
        (self.data / "frames.json").write_text("[]")
        with self.assertRaises(ValueError):
            Store(self.data, self.state)

    def test_http_auth_blinding_and_save(self):
        server = make_server(self.store, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/state") as response:
                state = json.load(response)
            payload = json.dumps(dict(id=state["items"][0]["id"], label="ok", revision=0)).encode()
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + "/api/review", data=payload, headers={"Content-Type": "application/json"}))
            self.assertEqual(error.exception.code, 403)
            with urlopen(Request(base + "/api/review", data=payload, headers={"Content-Type": "application/json", "X-Review-Token": state["token"]})) as response:
                self.assertEqual(json.load(response)["progress"]["human"]["done"], 1)
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/api/results")
            self.assertEqual(error.exception.code, 409)
            with self.assertRaises(HTTPError):
                urlopen(base + "/data/source_snapshot.json")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
