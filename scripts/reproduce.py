"""Single entry point for the reproducible experiment workflow (per dataset)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dataset_config import ROOT, expected_selections, spec


SCRIPTS = ROOT / "scripts"


def command(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *arguments]


class Workflow:
    def __init__(
        self, dataset: str, device: str | None, force: bool, dry_run: bool,
        accept_dataset_licenses: bool,
    ) -> None:
        self.dataset = dataset
        self.output = spec(dataset).output_dir
        self.device = device
        self.force = force
        self.dry_run = dry_run
        self.accept_dataset_licenses = accept_dataset_licenses

    def run(self, title: str, args: list[str]) -> None:
        printable = " ".join(args)
        print(f"\n== {title} ==\n{printable}", flush=True)
        if not self.dry_run:
            subprocess.run(args, cwd=ROOT, check=True)

    def dataset_command(self, script: str, *arguments: str) -> list[str]:
        return command(script, "--dataset", self.dataset, *arguments)

    def download(self) -> None:
        args = self.dataset_command("download_datasets.py")
        if self.accept_dataset_licenses:
            args.append("--accept-license")
        if self.force:
            args.append("--force")
        if self.dry_run:
            args.append("--dry-run")
        self.run(f"Obter dataset {self.dataset}", args)

    def split(self) -> None:
        if (self.output / "split.json").exists() and not self.force:
            print("\n== Split ==\nreutilizando partições congeladas")
            return
        args = self.dataset_command("generate_split.py")
        if self.force:
            args.append("--force")
        self.run("Gerar partições fixas (semente 42)", args)

    def prepare(self) -> None:
        args = self.dataset_command("prepare_bvtsld_yolo.py")
        if self.force:
            args.append("--force")
        self.run("Materializar dataset YOLO", args)

    def embeddings(self) -> None:
        dataset = spec(self.dataset)
        args = self.dataset_command("generate_embeddings.py")
        if self.device:
            args.extend(["--device", self.device])
        if dataset.embeddings_path.exists() and dataset.patterns_path.exists() and not self.force:
            args.extend(["--verify", "--sample", "32"])
            title = "Verificar representações congeladas"
        else:
            title = "Gerar representações DINO"
        self.run(title, args)

    def selections(self) -> None:
        paths = list((self.output / "selections").glob("*.json"))
        if len(paths) == expected_selections() and not self.force:
            print(f"\n== Seleções ==\nreutilizando {len(paths)} seleções versionadas")
            return
        self.run(
            "Gerar seleções", self.dataset_command("generate_bvtsld_local_selections.py")
        )

    def oracle(self) -> None:
        checkpoint = self.output / "runs" / "oracle" / "weights" / "best.pt"
        result = self.output / "oracle_results.json"
        if checkpoint.exists() and result.exists() and not self.force:
            print("\n== Oráculo ==\nreutilizando checkpoint local validado")
            return
        args = self.dataset_command("train_oracle.py")
        if self.device:
            args.extend(["--device", self.device])
        self.run("Treinar oráculo (100% do pool)", args)

    def smoke(self) -> None:
        args = self.dataset_command("run_local_triage.py", "--smoke")
        if self.device:
            args.extend(["--device", self.device])
        self.run("Smoke test (2 épocas)", args)

    def audit(self) -> None:
        if self.dataset != "bvtsld":
            print(f"\n== Auditoria ==\nsem auditor bruto para {self.dataset} ainda")
            return
        self.run("Auditoria", command("audit_bvtsld_local_pretrain.py"))

    def verify(self) -> None:
        self.run("Validação dos artefatos", self.dataset_command("validate_bvtsld.py"))

    def train(self) -> None:
        args = self.dataset_command("run_local_triage.py", "--isolate")
        if self.device:
            args.extend(["--device", self.device])
        self.run("Grade completa (328 runs, retomável, 1 processo por run)", args)

    def analyze(self) -> None:
        results = self.output / "triage_results.csv"
        if not results.exists() and not self.dry_run:
            raise FileNotFoundError(f"complete a grade antes da análise: {results}")
        self.run(
            "Análise estatística",
            command(
                "analyze_triage.py", str(results), "--output",
                str(self.output / "triage_analysis.csv"),
            ),
        )
        self.run("Resumo de métricas", self.dataset_command("summarize_metrics.py"))

    def quick(self) -> None:
        self.download()
        self.split()
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
    parser.add_argument("--dataset", default="bvtsld", choices=("bvtsld", "tt100k"))
    parser.add_argument(
        "--stage", required=True,
        choices=(
            "download", "split", "prepare", "embeddings", "selections", "oracle", "smoke",
            "audit", "verify", "train", "analyze", "quick", "all",
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
        args.dataset, args.device, args.force, args.dry_run, args.accept_dataset_licenses,
    )
    getattr(workflow, args.stage)()


if __name__ == "__main__":
    main()
