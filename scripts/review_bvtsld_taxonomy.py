"""Local browser UI for a resumable code-level review of the BVTSLD taxonomy."""
from __future__ import annotations

import argparse
import io
import json
import os
import threading
import webbrowser
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    ROOT / "datasets" / "bvtsld" / "Brazilian Vertical Traffic Signs and Lights Dataset"
)
DEFAULT_REVIEW_FILE = ROOT / "outputs" / "bvtsld" / "taxonomy_human_review.json"
WEB_ROOT = ROOT / "tools" / "taxonomy_review"
MAP_VERSION = "bvtsld-code-review-v2"
TARGET_CLASSES = ["regulatory", "warning", "information"]
CODE_MAP = {
    "000000": "regulatory",
    "000001": "regulatory",
    "000003": "regulatory",
    "000004": "regulatory",
    "000007": "regulatory",
    "000008": "regulatory",
    "000009": "regulatory",
    "000023": "regulatory",
    "000028": "regulatory",
    "000042": "regulatory",
    "000025": "warning",
    "000035": "information",
    "000040": "information",
    "000051": "quarantine_traffic_light",
    "000052": "quarantine_traffic_light",
    "000053": "quarantine_traffic_light",
}
CODE_REFERENCE = {
    "000000": ("R-1", "Parada obrigatória"),
    "000001": ("R-2", "Dê a preferência"),
    "000003": ("R-4a", "Proibido virar à esquerda"),
    "000004": ("R-4b", "Proibido virar à direita"),
    "000007": ("R-6a", "Proibido estacionar"),
    "000008": ("R-6b", "Estacionamento regulamentado"),
    "000009": ("R-6c", "Proibido parar e estacionar"),
    "000023": ("R-19", "Velocidade máxima permitida"),
    "000025": ("A-18", "Saliência ou lombada"),
    "000028": ("R-24a", "Sentido de circulação da via/pista"),
    "000035": (
        "R-27",
        "Ônibus, caminhões e veículos de grande porte mantenham-se à direita",
    ),
    "000040": ("R-32", "Circulação exclusiva de ônibus"),
    "000042": ("R-12", "Proibido trânsito de bicicletas"),
    "000051": ("Semáforo", "Foco amarelo"),
    "000052": ("Semáforo", "Foco vermelho"),
    "000053": ("Semáforo", "Foco verde"),
}
DECISIONS = {"approve", "remap", "mixed", "quarantine", "ambiguous"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_items(dataset_root: Path) -> tuple[list[dict], dict[str, Path]]:
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    items: list[dict] = []
    image_paths: dict[str, Path] = {}
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        if "@" in xml_path.stem:
            continue
        root = ET.parse(xml_path).getroot()
        width = int(float(root.findtext(".//size/width") or 800))
        height = int(float(root.findtext(".//size/height") or 450))
        image_path = images_dir / f"{xml_path.stem}.jpg"
        if not image_path.exists():
            continue
        image_paths[xml_path.stem] = image_path.resolve()
        for box_index, obj in enumerate(root.findall(".//object"), 1):
            code = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = max(0, int(float(box.findtext("xmin") or 0)))
            ymin = max(0, int(float(box.findtext("ymin") or 0)))
            xmax = min(width, int(float(box.findtext("xmax") or width)))
            ymax = min(height, int(float(box.findtext("ymax") or height)))
            item_id = f"{xml_path.stem}:{box_index:04d}"
            items.append(
                {
                    "item_id": item_id,
                    "image_id": xml_path.stem,
                    "box_index": box_index,
                    "source_code": code,
                    "bbox_xyxy": [xmin, ymin, xmax, ymax],
                    "image_width": width,
                    "image_height": height,
                    "crop_url": f"/api/crop/{item_id}",
                    "image_url": f"/api/image/{xml_path.stem}",
                }
            )
    return items, image_paths


def new_review_state(codes: list[str]) -> dict:
    now = utc_now()
    return {
        "schema_version": 2,
        "dataset": "bvtsld",
        "map_version": MAP_VERSION,
        "review": {
            "status": "in_progress",
            "reviewers": [],
            "scope": "source_codes",
            "review_unit": "source_code",
            "started_at_utc": now,
            "updated_at_utc": now,
            "completed_at_utc": None,
        },
        "code_map_reviewed": {code: CODE_MAP.get(code, "unmapped") for code in codes},
        "decisions": {},
        "summary": {"total": len(codes), "reviewed": 0, "remaining": len(codes)},
    }


def summarize(state: dict, codes: list[str], items: list[dict]) -> dict:
    decisions = state.get("decisions", {})
    by_decision = Counter()
    for code in codes:
        record = decisions.get(code)
        by_decision[record.get("decision") if record else "pending"] += 1
    reviewed = sum(code in decisions for code in codes)
    occurrences = Counter(item["source_code"] for item in items)
    return {
        "total": len(codes),
        "reviewed": reviewed,
        "remaining": len(codes) - reviewed,
        "unit": "source_code",
        "total_occurrences": len(items),
        "occurrences_by_code": dict(sorted(occurrences.items())),
        "by_decision": dict(sorted(by_decision.items())),
    }


class ReviewStore:
    def __init__(self, path: Path, items: list[dict]):
        self.path = path
        self.items = items
        self.codes = sorted({item["source_code"] for item in items})
        self.items_by_id = {item["item_id"]: item for item in items}
        self.items_by_code = {
            code: [item for item in items if item["source_code"] == code]
            for code in self.codes
        }
        self.lock = threading.Lock()
        if path.exists():
            self.state = json.loads(path.read_text())
            if self.state.get("dataset") != "bvtsld":
                raise ValueError(f"unexpected dataset in {path}")
            if self.state.get("map_version") != MAP_VERSION:
                raise ValueError(
                    f"review file uses {self.state.get('map_version')!r}; "
                    f"move it aside before starting the {MAP_VERSION!r} code-level review"
                )
        else:
            self.state = new_review_state(self.codes)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        self.state["summary"] = summarize(self.state, self.codes, self.items)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, self.path)

    def public_state(self) -> dict:
        with self.lock:
            self._refresh_summary()
            codes = []
            for code in self.codes:
                official_code, expected_name = CODE_REFERENCE.get(
                    code, ("Sem referência", "Código não documentado na interface")
                )
                codes.append(
                    {
                        "source_code": code,
                        "official_code": official_code,
                        "expected_name": expected_name,
                        "current_group": CODE_MAP.get(code, "unmapped"),
                        "occurrence_count": len(self.items_by_code[code]),
                        "occurrences": self.items_by_code[code],
                    }
                )
            return {
                "dataset": self.state["dataset"],
                "map_version": self.state["map_version"],
                "review": self.state["review"],
                "summary": self.state["summary"],
                "decisions": self.state.get("decisions", {}),
                "allowed_decisions": sorted(DECISIONS),
                "target_classes": TARGET_CLASSES,
                "codes": codes,
                "cheat_sheets": [
                    {
                        "title": "Tipos de placas de trânsito",
                        "image_url": "/assets/tipos-de-placas-de-transito.png?v=1",
                    }
                ],
            }

    def set_reviewer(self, reviewer: str) -> dict:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        with self.lock:
            reviewers = self.state["review"].setdefault("reviewers", [])
            if reviewer not in reviewers:
                reviewers.append(reviewer)
            self.state["review"]["updated_at_utc"] = utc_now()
            self._write()
            return self.state["review"]

    def save_decision(self, payload: dict) -> dict:
        source_code = str(payload.get("source_code", ""))
        decision = str(payload.get("decision", ""))
        reviewer = str(payload.get("reviewer", "")).strip()
        corrected_class = payload.get("corrected_class") or None
        note = str(payload.get("note", "")).strip()
        if source_code not in self.codes:
            raise ValueError("unknown source_code")
        if decision not in DECISIONS:
            raise ValueError("invalid decision")
        if not reviewer:
            raise ValueError("reviewer is required")
        if decision == "remap" and corrected_class not in TARGET_CLASSES:
            raise ValueError("corrected_class is required for remap")
        if decision != "remap":
            corrected_class = None
        with self.lock:
            reviewers = self.state["review"].setdefault("reviewers", [])
            if reviewer not in reviewers:
                reviewers.append(reviewer)
            self.state.setdefault("decisions", {})[source_code] = {
                "decision": decision,
                "corrected_class": corrected_class,
                "note": note,
                "reviewer": reviewer,
                "reviewed_at_utc": utc_now(),
                "occurrences_inspected": len(self.items_by_code[source_code]),
            }
            self.state["review"]["status"] = "in_progress"
            self.state["review"]["completed_at_utc"] = None
            self.state["review"]["updated_at_utc"] = utc_now()
            self._refresh_summary()
            self._write()
            return {
                "record": self.state["decisions"][source_code],
                "summary": self.state["summary"],
            }

    def finalize(self, reviewer: str) -> dict:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        with self.lock:
            self._refresh_summary()
            if self.state["summary"]["remaining"]:
                raise ValueError("all source codes must be reviewed before finalization")
            reviewers = self.state["review"].setdefault("reviewers", [])
            if reviewer not in reviewers:
                reviewers.append(reviewer)
            now = utc_now()
            self.state["review"].update(
                {
                    "status": "human_approved",
                    "updated_at_utc": now,
                    "completed_at_utc": now,
                }
            )
            self._write()
            return self.state["review"]


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, handler, store, image_paths):
        super().__init__(address, handler)
        self.store = store
        self.image_paths = image_paths


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_bytes(data, "application/json; charset=utf-8", status, "no-store")

    def send_file(self, path: Path, content_type: str, cache: str = "no-store") -> None:
        self.send_bytes(path.read_bytes(), content_type, HTTPStatus.OK, cache)

    def send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        cache: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def send_crop(self, item_id: str) -> None:
        item = self.server.store.items_by_id.get(item_id)
        if not item:
            self.send_json({"error": "crop not found"}, HTTPStatus.NOT_FOUND)
            return
        image_path = self.server.image_paths[item["image_id"]]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            x1, y1, x2, y2 = item["bbox_xyxy"]
            width, height = max(1, x2 - x1), max(1, y2 - y1)
            padding = int(max(width, height) * 0.28)
            crop = image.crop(
                (
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(image.width, x2 + padding),
                    min(image.height, y2 + padding),
                )
            )
            crop.thumbnail((300, 220))
            output = io.BytesIO()
            crop.save(output, format="JPEG", quality=86)
        self.send_bytes(
            output.getvalue(), "image/jpeg", cache="public, max-age=3600, immutable"
        )

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/state":
            self.send_json(self.server.store.public_state())
            return
        if path.startswith("/api/image/"):
            image_id = path.removeprefix("/api/image/")
            image_path = self.server.image_paths.get(image_id)
            if image_path and image_path.exists():
                self.send_file(image_path, "image/jpeg", "public, max-age=3600")
            else:
                self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
            return
        if path.startswith("/api/crop/"):
            self.send_crop(path.removeprefix("/api/crop/"))
            return
        static_files = {
            "/": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
            "/app.js": (WEB_ROOT / "app.js", "text/javascript; charset=utf-8"),
            "/styles.css": (WEB_ROOT / "styles.css", "text/css; charset=utf-8"),
            "/assets/tipos-de-placas-de-transito.png": (
                WEB_ROOT / "assets" / "tipos-de-placas-de-transito.png",
                "image/png",
            ),
        }
        if path in static_files:
            self.send_file(*static_files[path], cache="public, max-age=3600")
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 100_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            if self.path == "/api/reviewer":
                result = self.server.store.set_reviewer(str(payload.get("reviewer", "")))
            elif self.path == "/api/review":
                result = self.server.store.save_decision(payload)
            elif self.path == "/api/finalize":
                result = self.server.store.finalize(str(payload.get("reviewer", "")))
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(result)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    required = [
        WEB_ROOT / "index.html",
        WEB_ROOT / "app.js",
        WEB_ROOT / "styles.css",
        WEB_ROOT / "assets" / "tipos-de-placas-de-transito.png",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    items, image_paths = load_items(args.dataset_root)
    if not items:
        raise RuntimeError("no original bounding boxes found")
    store = ReviewStore(args.review_file, items)
    server = ReviewServer((args.host, args.port), Handler, store, image_paths)
    url = f"http://{args.host}:{args.port}"
    print(
        f"BVTSLD taxonomy reviewer: {len(store.codes)} source codes, "
        f"{len(items)} occurrences"
    )
    print(f"Review file: {args.review_file}")
    print(f"Open: {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
