# slm-forge

A lightweight, reproducible fine-tuning and evaluation harness for small language models (SLMs), built for low-resource multilingual settings (Darija, French, English) — with **factual reliability**, not just loss or perplexity, as a first-class evaluation metric.

> **Status: pre-alpha (milestone M0).** The package skeleton, config and dataset schemas, the outcome decision table and `slm-forge data validate` work today; scoring, generation and training land in M1–M3. Commands that are not implemented yet exit with code 5 and name their milestone. See [ROADMAP.md](ROADMAP.md).

```
slm-forge finetune --config configs/qwen2.5-7b-darija.yaml
slm-forge eval --checkpoint runs/qwen2.5-7b-darija/checkpoint-500 --suite factual-reliability
```

---

## Why

Fine-tuning frameworks such as Axolotl, LLaMA-Factory and Unsloth are general-purpose: they report training loss, perplexity and, through tools like lm-evaluation-harness, accuracy on mostly English-first benchmarks. Hallucination benchmarks exist (TruthfulQA and others), but they are run as one-off evaluations, not wired into the fine-tuning loop. slm-forge combines the two: it tracks **confabulation** (a fine-tuned model confidently stating something false) at every checkpoint, against domain ground truth, in low-resource settings like Darija that standard benchmarks barely cover.

That gap matters most exactly where fine-tuning is riskiest: financial and legal domains, where a model that sounds fluent but invents a figure or a clause is worse than a model that says "I don't know." slm-forge exists to make that failure mode measurable and trackable throughout training, not discovered after deployment.

---

## What it does

- Fine-tunes open-weight SLMs with LoRA/QLoRA (Unsloth on NVIDIA GPUs, plain HF + PEFT everywhere else) on your task-specific data.
- Runs a built-in **factual-reliability eval suite** against held-out ground-truth claims at every checkpoint — not just at the end of training.
- Tracks confabulation rate, factual accuracy, and standard task metrics side by side, across the training run, so you can see when a checkpoint starts trading reliability for fluency.
- Supports low-resource multilingual settings (Darija, French, English) as first-class targets, not an afterthought bolted onto an English-first pipeline.

---

## Architecture

```
   training_data/*.jsonl                         data/eval/ground_truth.jsonl
            │                                                 │
            ▼                                                 │
   ┌──────────────────────┐                                   │
   │ data loader          │  assistant-mask labels,           │
   │ + contamination guard│  language resampling              │
   └──────────┬───────────┘                                   │
              ▼                                               │
   ┌──────────────────────┐                                   │
   │ LoRA / QLoRA         │  Unsloth (CUDA, subprocess)       │
   │ fine-tuning          │  or HF + PEFT (CPU/MPS/CUDA)      │
   └──────────┬───────────┘                                   │
              │ checkpoint-N/ saved                           │
              ▼                                               ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ generation (greedy, teacher-forced logprobs) → generations.jsonl │
   └──────────────────────────────┬───────────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ scoring (torch-free): normalize → abstention/hedge cues →        │
   │ extract candidates → match → confidence → one outcome per item   │
   └──────────────────────────────┬───────────────────────────────────┘
                                  ▼
              per-checkpoint metrics.json (CIs, buckets, provenance)
                                  ▼
              run_metrics.json: trend, paired tests vs base, flags
                                  ▼
              report.md / report.html
```

Fine-tuning and evaluation are decoupled: any checkpoint can be re-evaluated independently, and the eval suite runs identically whether the checkpoint came from slm-forge or another pipeline. Scoring needs only `{"item_id", "response"}` lines (`slm-forge score`), and re-scoring never regenerates (`slm-forge rescore`). Details: [docs/architecture.md](docs/architecture.md).

---

## Models

| Role | Examples | Why |
|---|---|---|
| Base SLMs supported | Phi-4-mini, Mistral-7B, Qwen2.5-7B, SmolLM2-1.7B | Range of sizes for different compute budgets |
| Fine-tuning backend | Unsloth (LoRA/QLoRA, NVIDIA CUDA) or HF + PEFT (LoRA on CPU/MPS/CUDA) | Unsloth is fast and memory-efficient on one GPU; HF + PEFT keeps tests and small runs hardware-independent |
| Teacher models (optional, distillation only) | Claude, GPT (API) | Used only to generate distillation data — never deployed as the served model |

All fine-tuned models run and are evaluated locally. Teacher-model API calls are optional and only used during data preparation, not at inference or eval time.

---

## Requirements

- Python 3.11+
- For scoring, validation and reports: nothing beyond the core install (no GPU, no torch).
- For QLoRA on 7B-class models: an NVIDIA GPU with 16 GB+ VRAM and [Unsloth](https://github.com/unslothai/unsloth) (`.[hf,qlora,unsloth]`).
- For LoRA without CUDA (CPU, Apple Silicon via MPS, best effort): the HF + PEFT backend (`.[hf]`); QLoRA needs CUDA.

---

## Installation

```bash
git clone https://github.com/TAJANIYoussef/slm-forge.git
cd slm-forge

uv pip install -e ".[dev]"            # torch-free core: validate, score, rescore, report
uv pip install -e ".[dev,hf]"         # + generation and HF/PEFT LoRA training
uv pip install -e ".[dev,hf,qlora,unsloth]"   # + QLoRA via Unsloth (NVIDIA CUDA)
```

---

## Usage

### Fine-tune a model

```bash
slm-forge finetune --config configs/qwen2.5-7b-darija.yaml
```

Checkpoints and training logs are written to `runs/<run-name>/`.

### Evaluate a checkpoint

```bash
slm-forge eval \
  --checkpoint runs/qwen2.5-7b-darija/checkpoint-500 \
  --suite factual-reliability \
  --ground-truth data/eval/ground_truth.jsonl
```

### Evaluate an entire training run (all checkpoints)

```bash
slm-forge eval-run --run runs/qwen2.5-7b-darija --suite factual-reliability
```

This produces a per-checkpoint trend report, so you can see whether factual reliability degrades as training accuracy improves — a trade-off standard fine-tuning metrics don't surface.

---

## Evaluation suite

Every item gets exactly one outcome from a pure decision table (full spec: [docs/metrics.md](docs/metrics.md)). The three headline metrics are:

| Metric | Definition | Denominator |
|---|---|---|
| Factual accuracy | Answer matches ground truth (hedged or not) | all answerable items |
| Confabulation rate | Answer is wrong, **not hedged**, and the model's answer-span confidence is ≥ τ | all scored items |
| Abstention rate | Model declines or rejects the premise on an *unanswerable* item | all unanswerable items |

These are reported alongside them:
- `unhedged_error_rate`: the τ-free version of confabulation rate, comparable across models.
- `over_abstention_rate`: the model declines answerable items.
- `invalid_rate` and `unresolved_rate`. INVALID and UNRESOLVED items are never dropped from denominators. Items the rules cannot decide are reported as best/worst-case bounds.

Everything is broken down by language/script bucket (`darija_arab`, `darija_latn`, `fr`, `en`). The headline is the macro average over those buckets.

### Labeling format

Add entries to `data/eval/ground_truth.jsonl`, one JSON object per line. The examples use fictional facts:

```jsonl
{"id": "fin-fr-0001", "fact_id": "fin-atlas-ltv", "prompt": "Quel est le ratio LTV maximal du Prêt Atlas Habitat ?", "language": "fr", "domain": "finance", "expected_answer": "70%", "answer_type": "percent", "numeric": {"value": 70, "unit": "%"}, "aliases": ["70 %", "soixante-dix pour cent"], "known_false": ["80%"]}
{"id": "fin-dar-l-0001", "fact_id": "fin-atlas-ltv", "prompt": "ch7al howa l'LTV l9ousswa dyal Prêt Atlas Habitat?", "language": "darija", "script": "latn", "domain": "finance", "expected_answer": "70%", "answer_type": "percent", "numeric": {"value": 70, "unit": "%"}, "aliases": ["sb3in f lmya", "سبعين فالمية"]}
{"id": "leg-en-u-0001", "prompt": "What does Article 912 of the Atlas Family Code say about alimony?", "language": "en", "domain": "legal", "expected_answer": null, "unanswerable_reason": "nonexistent_entity"}
```

Core fields:

| Field | Required? | Rules |
|---|---|---|
| `id` | required | unique |
| `prompt` | required | — |
| `language` | required | `darija`, `fr` or `en` |
| `script` | required for Darija | `arab`, `latn` or `mixed` |
| `expected_answer` | **required key** | `null` means the item is unanswerable, and then `unanswerable_reason` is required |
| `fact_id` | optional | shared by translations of the same fact |
| `answer_type` with `numeric` / `date` / `citation` | optional | typed matching with tolerances |
| `aliases` / `known_false` | optional | — |
| `allow_abstention` | optional, default `true` | answerable items only; `false` makes "I don't know" a penalized ABSTAIN_UNWARRANTED |

The README's original 5-field records are still accepted and upgraded automatically. Validate a file with:

```bash
slm-forge data validate data/eval/ground_truth.jsonl
```

Full reference and validation rules: [docs/dataset.md](docs/dataset.md).

---

## Project structure

```
slm-forge/
├── src/slmforge/
│   ├── cli.py                    # Entry point: slm-forge
│   ├── config/
│   │   └── schema.py             # Training/eval config schema
│   ├── data/
│   │   ├── loader.py             # Dataset loading + formatting
│   │   └── distill.py            # Optional teacher-model distillation data prep
│   ├── train/
│   │   └── lora.py               # LoRA/QLoRA fine-tuning (Unsloth backend)
│   ├── eval/
│   │   ├── dataset.py            # ground_truth.jsonl loader
│   │   ├── factual_reliability.py  # Confabulation/accuracy/abstention scoring
│   │   ├── task_metrics.py       # Standard task metrics (accuracy, F1, BLEU/ROUGE)
│   │   └── runner.py             # Per-checkpoint and per-run eval orchestration
│   └── report/
│       └── render.py             # Markdown/HTML report generation
├── configs/                      # Example training configs
├── tests/
│   ├── unit/
│   └── integration/
└── data/
    └── eval/
        └── ground_truth.jsonl    # Labeled factual-reliability eval set
```

---

## Limitations

- **7B-class models with QLoRA need an NVIDIA GPU with ~16 GB VRAM**; without CUDA, use a smaller base model with LoRA on the HF + PEFT backend.
- **Abstention and hedge detection is lexicon-based**; it is validated per language/script against human labels, and undecidable cases are reported as UNRESOLVED bounds rather than guessed.
- **Ground-truth eval sets must be hand-labeled** per domain and language — the suite doesn't auto-generate them.
- **Darija support depends on base-model tokenizer coverage**; some SLMs handle Darija (especially Arabizi/Latin-script Darija) better than others.
- **Distillation from teacher models is optional and one-time** — it is not required to use the fine-tuning or evaluation pipeline. Check the provider's terms first ([docs/distillation-terms.md](docs/distillation-terms.md)).

---

## License

MIT