"""``slm-forge`` command-line interface.

Every command imports its implementation lazily, so ``slm-forge --help`` and the
torch-free commands (``data``, ``score``, ``rescore``, ``report``) never load torch.
The training backend is resolved before transformers is imported, because Unsloth
patches transformers/peft on import.

Exit codes: 0 ok, 1 data/config/protocol validation error, 2 usage, 3 environment
(CUDA, 4-bit or an offline model unavailable), 4 quality gate failed, 5 not implemented yet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from slmforge import __version__

EXIT_VALIDATION, EXIT_USAGE, EXIT_ENV, EXIT_GATE, EXIT_NOT_IMPLEMENTED = 1, 2, 3, 4, 5

app = typer.Typer(
    name="slm-forge",
    help="Fine-tune small language models and track factual reliability at every checkpoint.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)
data_app = typer.Typer(help="Validate and inspect ground-truth datasets.", no_args_is_help=True)
protocol_app = typer.Typer(help="Show or lock the evaluation protocol (pre-registration).", no_args_is_help=True)
review_app = typer.Typer(help="Human review loop for UNRESOLVED items.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(protocol_app, name="protocol")
app.add_typer(review_app, name="review")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config (run or eval).")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Override: dotted.key=value (repeatable).")]
SuiteOpt = Annotated[str, typer.Option("--suite", help="Evaluation suite.")]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"slm-forge {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """slm-forge: fine-tuning + factual-reliability evaluation for small LMs (Darija, French, English)."""


# --------------------------------------------------------------------------- training


@app.command()
def finetune(
    config: Annotated[Path, typer.Option("--config", "-c", help="Run config YAML.")],
    set_: SetOpt = None,
    backend: Annotated[str | None, typer.Option(help="Override train.backend: auto | unsloth | hf_peft.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Validate config and data, print the plan, do not train.")] = False,
    resume: Annotated[bool, typer.Option(help="Resume from the latest checkpoint.")] = False,
    overwrite: Annotated[bool, typer.Option(help="Overwrite an existing run directory.")] = False,
    skip_eval: Annotated[bool, typer.Option(help="Do not run eval-run after training.")] = False,
) -> None:
    """Fine-tune a model with LoRA/QLoRA; checkpoints go to runs/<run.name>/."""
    from slmforge.config.load import load_root_config

    overrides = list(set_ or [])
    if backend:
        overrides.append(f"train.backend={backend}")
    cfg = load_root_config(config, overrides)
    from slmforge.train.lora import finetune as run_finetune

    result = run_finetune(cfg, dry_run=dry_run, resume=resume, overwrite=overwrite, skip_eval=skip_eval)
    typer.echo(f"run: {result.run_dir}")


# --------------------------------------------------------------------------- evaluation


def _eval_config(config: Path | None, set_: list[str] | None, ground_truth: Path | None):  # noqa: ANN202
    from slmforge.config.load import load_eval_config

    if config is None and ground_truth is None:
        raise typer.BadParameter("pass --ground-truth or a --config with eval.ground_truth")
    return load_eval_config(config, list(set_ or []), ground_truth=ground_truth)


@app.command("eval")
def eval_cmd(
    checkpoint: Annotated[str, typer.Option(help="Checkpoint dir (adapter or full model) or hub id[@revision].")],
    suite: SuiteOpt = "factual-reliability",
    ground_truth: Annotated[Path | None, typer.Option(help="Ground-truth JSONL.")] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    base_model: Annotated[str | None, typer.Option(help="Base model for an adapter checkpoint.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output dir (default eval_out/<checkpoint>/...).")] = None,
) -> None:
    """Evaluate one checkpoint on the factual-reliability suite."""
    cfg = _eval_config(config, [*(set_ or []), f"suite=[{suite}]"], ground_truth)
    from slmforge.eval.checkpoints import resolve_checkpoint
    from slmforge.eval.runner import EvalRunner

    ckpt = resolve_checkpoint(checkpoint, base_model=base_model)
    out_dir = out or Path("eval_out") / ckpt.name / suite
    metrics = EvalRunner(cfg).eval_checkpoint(ckpt, out_dir=out_dir)
    typer.echo(metrics.model_dump_json(indent=2))


@app.command("eval-run")
def eval_run(
    run: Annotated[Path, typer.Option(help="Run directory (runs/<name>).")],
    suite: SuiteOpt = "factual-reliability",
    config: ConfigOpt = None,
    set_: SetOpt = None,
    steps: Annotated[str | None, typer.Option(help="Comma-separated steps to evaluate (default: schedule).")] = None,
    resume: Annotated[bool, typer.Option(help="Reuse cached generations.")] = True,
) -> None:
    """Evaluate every scheduled checkpoint of a run and write the trend report."""
    cfg_path = config or run / "resolved_config.yaml"
    cfg = _eval_config(cfg_path, [*(set_ or []), f"suite=[{suite}]"], None)
    from slmforge.eval.runner import EvalRunner

    step_list = [int(s) for s in steps.split(",")] if steps else None
    EvalRunner(cfg).eval_run(run, steps=step_list, resume=resume)


@app.command()
def score(
    responses: Annotated[Path, typer.Option(help="JSONL of {item_id, response} from any pipeline.")],
    ground_truth: Annotated[Path, typer.Option(help="Ground-truth JSONL.")],
    out: Annotated[Path, typer.Option(help="Output directory.")],
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Score external responses (no model, no torch)."""
    cfg = _eval_config(config, set_, ground_truth)
    from slmforge.eval.runner import EvalRunner

    metrics = EvalRunner(cfg).score_external(responses, out_dir=out)
    typer.echo(metrics.model_dump_json(indent=2))


@app.command()
def rescore(
    eval_dir: Annotated[Path, typer.Argument(help="An eval output dir containing generations.jsonl.")],
    config: ConfigOpt = None,
    tau: Annotated[float | None, typer.Option(help="Confidence threshold override.")] = None,
) -> None:
    """Re-run scoring from cached generations (never regenerates)."""
    cfg = _eval_config(config or eval_dir / "eval_config.yaml", None, None)
    from slmforge.eval.runner import EvalRunner

    EvalRunner(cfg).rescore(eval_dir, tau=tau)


@app.command()
def report(
    run: Annotated[Path, typer.Option(help="Run directory.")],
    fmt: Annotated[str, typer.Option("--format", help="md | html | both")] = "both",
) -> None:
    """Render report.md / report.html for a run."""
    from slmforge.report.render import render_run_report

    formats = ("md", "html") if fmt == "both" else (fmt,)
    for p in render_run_report(run, formats=formats):
        typer.echo(str(p))


# --------------------------------------------------------------------------- data


@data_app.command("validate")
def data_validate(
    path: Annotated[Path, typer.Argument(help="Ground-truth JSONL.")],
    strict: Annotated[bool, typer.Option(help="Treat warnings as errors.")] = False,
) -> None:
    """Validate a ground-truth file (rules E01-E14, W01-W13; see docs/dataset.md)."""
    from slmforge.eval.dataset import GroundTruthError, load_ground_truth

    try:
        gt = load_ground_truth(path, strict=strict)
    except GroundTruthError as e:
        for issue in e.issues:
            typer.echo(str(issue), err=True)
        raise typer.Exit(EXIT_VALIDATION) from None
    for issue in gt.issues:
        typer.echo(str(issue), err=True)
    n_unans = sum(1 for i in gt.items if not i.answerable)
    typer.echo(f"ok: {len(gt)} items ({n_unans} unanswerable), sha256 {gt.sha256[:12]}")
    for (bucket, domain, answerable), n in sorted(gt.coverage_table().items()):
        typer.echo(f"  {bucket:<13} {domain:<10} {'answerable' if answerable else 'unanswerable':<13} {n}")


@data_app.command("stats")
def data_stats(path: Annotated[Path, typer.Argument(help="Ground-truth JSONL.")]) -> None:
    """Coverage, per-stratum CI half-widths, design effect and minimum detectable effect."""
    from slmforge.eval.dataset import dataset_stats, load_ground_truth

    typer.echo(dataset_stats(load_ground_truth(path)))


@data_app.command("contamination")
def data_contamination(
    train: Annotated[list[Path], typer.Option(help="Training JSONL file(s).")],
    ground_truth: Annotated[Path, typer.Option(help="Ground-truth JSONL.")],
) -> None:
    """Report overlap between training data and eval prompts/facts."""
    from slmforge.data.contamination import TrainIndex, contamination_report
    from slmforge.data.loader import load_train_records
    from slmforge.eval.dataset import load_ground_truth
    from slmforge.config.schema import DataSpec

    records = load_train_records([DataSpec(path=p) for p in train], seed=0)
    report_ = contamination_report(load_ground_truth(ground_truth), TrainIndex.build(records))
    flagged = [r for r in report_.values() if r.contaminated]
    typer.echo(f"{len(flagged)} of {len(report_)} items flagged")


# --------------------------------------------------------------------------- protocol / review / validation


@protocol_app.command("show")
def protocol_show(config: Annotated[Path, typer.Option("--config", "-c")]) -> None:
    """Print the live protocol and its hash."""
    from slmforge.eval.dataset import load_ground_truth
    from slmforge.protocol import build_protocol

    cfg = _eval_config(config, None, None)
    p = build_protocol(cfg, load_ground_truth(cfg.ground_truth))
    typer.echo(p.model_dump_json(indent=2))
    typer.echo(f"sha256: {p.sha256()}")


@protocol_app.command("lock")
def protocol_lock(
    config: Annotated[Path, typer.Option("--config", "-c")],
    out: Annotated[Path, typer.Option(help="Lock file to write.")],
) -> None:
    """Write a protocol lock: later evals with a different protocol fail with ProtocolDrift."""
    from slmforge.eval.dataset import load_ground_truth
    from slmforge.protocol import build_protocol, lock_protocol

    cfg = _eval_config(config, None, None)
    lock_protocol(build_protocol(cfg, load_ground_truth(cfg.ground_truth)), out)
    typer.echo(f"locked: {out}")


@review_app.command("export")
def review_export(
    eval_dir: Annotated[list[Path], typer.Option(help="Eval output dir(s).")],
    out: Annotated[Path, typer.Option(help="Review queue JSONL to write.")],
) -> None:
    """Export UNRESOLVED and borderline items for human labelling."""
    from slmforge.eval.review import export_review_queue

    typer.echo(f"{export_review_queue(eval_dir, out)} items exported to {out}")


@review_app.command("import")
def review_import(
    path: Annotated[Path, typer.Argument(help="Labelled review JSONL.")],
    annotator: Annotated[str, typer.Option(help="Annotator id.")],
    cache: Annotated[Path, typer.Option(help="Verdict cache.")] = Path("runs/_shared/verdicts.sqlite"),
) -> None:
    """Import human labels into the verdict cache (used on the next rescore)."""
    from slmforge.eval.review import VerdictCache, import_reviews

    typer.echo(f"{import_reviews(path, VerdictCache(cache), annotator=annotator)} labels imported")


@app.command("validate-detectors")
def validate_detectors(
    labels: Annotated[Path, typer.Option(help="responses_gold.jsonl with human labels.")],
    config: ConfigOpt = None,
) -> None:
    """Check abstention/hedge detectors and the matcher against human labels (gates per stratum)."""
    from slmforge.eval.judge import validate_components

    cfg = _eval_config(config, None, None) if config else None
    report_ = validate_components(labels, cfg=cfg)  # type: ignore[call-arg]
    typer.echo(report_)
    if not all(c.passed for c in report_.components.values()):
        raise typer.Exit(EXIT_GATE)


@app.command("diagnose-tokenizer")
def diagnose_tokenizer(
    model: Annotated[list[str], typer.Option(help="Model id(s) or path(s).")],
    ground_truth: Annotated[Path, typer.Option(help="Ground-truth JSONL used as the text sample.")],
) -> None:
    """Tokenizer fertility, byte fallback and Arabizi digit handling per language/script bucket."""
    from slmforge.diagnostics.tokenizer import diagnose_tokenizer as run, texts_from_ground_truth
    from slmforge.eval.dataset import load_ground_truth

    texts = texts_from_ground_truth(load_ground_truth(ground_truth))
    for m in model:
        typer.echo(f"{m}: {run(m, texts)}")


@app.command()
def distill(config: Annotated[Path, typer.Option("--config", "-c", help="Distillation config YAML.")]) -> None:
    """Generate training data with a teacher model (optional; ToS-gated, see docs/distillation-terms.md)."""
    from slmforge.config.load import load_yaml
    from slmforge.config.schema import DistillConfig
    from slmforge.data.distill import run_distillation

    cfg = DistillConfig.model_validate(load_yaml(config))
    run_distillation(cfg, client=None, eval_guard=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- entry point


def main() -> None:
    """Console-script entry point: maps known exceptions to exit codes."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before any torch import
    from pydantic import ValidationError

    from slmforge._stub import NotImplementedYet

    try:
        app()
    except NotImplementedYet as e:
        typer.echo(f"not implemented yet: {e}", err=True)
        sys.exit(EXIT_NOT_IMPLEMENTED)
    except ValidationError as e:
        typer.echo(f"invalid config:\n{e}", err=True)
        sys.exit(EXIT_VALIDATION)
    except FileNotFoundError as e:
        typer.echo(f"file not found: {e.filename or e}", err=True)
        sys.exit(EXIT_VALIDATION)
    except OSError as e:  # EnvironmentError: CUDA/4-bit/offline model unavailable
        typer.echo(f"environment error: {e}", err=True)
        sys.exit(EXIT_ENV)


if __name__ == "__main__":
    main()
