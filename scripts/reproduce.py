"""Single entry point for the reproducible BVTSLD experiment workflow."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUT = ROOT / "outputs" / "bvtsld"
EXPECTED_SELECTIONS = 164


def command(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *arguments]


class Workflow:
    def __init__(
        self, device: str | None, force: bool, dry_run: bool,
        accept_dataset_licenses: bool,
    ) -> None:
        self.device = device
        self.force = force
        self.dry_run = dry_run
        self.accept_dataset_licenses = accept_dataset_licenses

    def run(self, title: str, args: list[str]) -> None:
        printable = " ".join(args)
        print(f"\n== {title} ==\n{printable}", flush=True)
        if not self.dry_run:
            subprocess.run(args, cwd=ROOT, check=True)

    def download(self) -> None:
        args = command("download_datasets.py", "--dataset", "bvtsld")
        if self.accept_dataset_licenses:
            args.append("--accept-license")
        if self.force:
            args.append("--force")
        if self.dry_run:
            args.append("--dry-run")
        self.run("Obter dataset BVTSLD", args)

    def prepare(self) -> None:
        args = command("prepare_bvtsld_yolo.py")
        if self.force:
            args.append("--force")
        self.run("Materializar dataset YOLO", args)

    def embeddings(self) -> None:
        global_path = OUTPUT / "embeddings_bvtsld_dinov2_full.npy"
        pattern_path = OUTPUT / "patterns_bvtsld_freesel.npz"
        args = command("generate_embeddings.py", "--dataset", "bvtsld")
        if self.device:
            args.extend(["--device", self.device])
        if global_path.exists() and pattern_path.exists() and not self.force:
            args.extend(["--verify", "--sample", "32"])
            title = "Verificar representações congeladas"
        else:
            title = "Gerar representações DINO"
        self.run(title, args)

    def selections(self) -> None:
        paths = list((OUTPUT / "selections").glob("*.json"))
        if len(paths) == EXPECTED_SELECTIONS and not self.force:
            print(f"\n== Seleções ==\nreutilizando {EXPECTED_SELECTIONS} seleções versionadas")
            return
        self.run("Gerar seleções", command("generate_bvtsld_local_selections.py"))

    def oracle(self) -> None:
        checkpoint = OUTPUT / "runs" / "oracle" / "weights" / "best.pt"
        result = OUTPUT / "oracle_results.json"
        if checkpoint.exists() and result.exists() and not self.force:
            print("\n== Oráculo ==\nreutilizando checkpoint local validado")
            return
        args = command("train_oracle.py")
        if self.device:
            args.extend(["--device", self.device])
        self.run("Treinar oráculo (100% do pool)", args)

    def smoke(self) -> None:
        args = command("run_local_triage.py", "--smoke")
        if self.device:
            args.extend(["--device", self.device])
        self.run("Smoke test (2 épocas)", args)

    def audit(self) -> None:
        self.run("Auditoria", command("audit_bvtsld_local_pretrain.py"))

    def verify(self) -> None:
        self.run("Validação dos artefatos", command("validate_bvtsld.py"))

    def train(self) -> None:
        args = command("run_local_triage.py")
        if self.device:
            args.extend(["--device", self.device])
        self.run("Grade completa (328 runs, retomável)", args)

    def analyze(self) -> None:
        results = OUTPUT / "triage_results.csv"
        if not results.exists() and not self.dry_run:
            raise FileNotFoundError(f"complete a grade antes da análise: {results}")
        self.run(
            "Análise estatística",
            command(
                "analyze_triage.py", str(results), "--output",
                str(OUTPUT / "triage_analysis.csv"),
            ),
        )
        self.run("Resumo de métricas", command("summarize_metrics.py"))

    def quick(self) -> None:
        self.download()
        self.prepare()
        self.embeddings()
        self.selections()
        self.smoke()
        self.audit()
        self.verify()

    def all(self) -> None:
        self.quick()
        self.oracle()
        self.train()
        self.analyze()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=(
            "download", "prepare", "embeddings", "selections", "oracle", "smoke", "audit",
            "verify", "train", "analyze", "quick", "all",
        ),
    )
    parser.add_argument("--device", help="Ultralytics/PyTorch device: cuda, mps or cpu")
    parser.add_argument("--force", action="store_true", help="Regenerate existing local artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument(
        "--accept-dataset-licenses", action="store_true",
        help="Confirm the dataset licenses when a download is necessary",
    )
    args = parser.parse_args()

    workflow = Workflow(
        args.device, args.force, args.dry_run, args.accept_dataset_licenses,
    )
    getattr(workflow, args.stage)()


if __name__ == "__main__":
    main()
