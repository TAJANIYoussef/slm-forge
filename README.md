# slm-forge

A lightweight, reproducible fine-tuning and evaluation harness for small language models (SLMs), built for low-resource multilingual settings (Darija, French, English) — with **factual reliability**, not just loss or perplexity, as a first-class evaluation metric.

```
slm-forge finetune --config configs/qwen2.5-7b-darija.yaml
slm-forge eval --checkpoint runs/qwen2.5-7b-darija/checkpoint-500 --suite factual-reliability
```

---

## Why

Most fine-tuning frameworks — Axolotl, LLaMA-Factory, Unsloth — are general-purpose and English-benchmark-centric. They optimize and report on training loss, perplexity, and downstream accuracy on English-first benchmarks. None of them treat **confabulation** (a fine-tuned model confidently stating something false) as a metric to track across checkpoints, and almost none of them evaluate low-resource multilingual settings like Darija at all.

That gap matters most exactly where fine-tuning is riskiest: financial and legal domains, where a model that sounds fluent but invents a figure or a clause is worse than a model that says "I don't know." slm-forge exists to make that failure mode measurable and trackable throughout training, not discovered after deployment.

---

## What it does

- Fine-tunes open-weight SLMs with LoRA/QLoRA (via Unsloth) on your task-specific data.
- Runs a built-in **factual-reliability eval suite** against held-out ground-truth claims at every checkpoint — not just at the end of training.
- Tracks confabulation rate, factual accuracy, and standard task metrics side by side, across the training run, so you can see when a checkpoint starts trading reliability for fluency.
- Supports low-resource multilingual settings (Darija, French, English) as first-class targets, not an afterthought bolted onto an English-first pipeline.

---

## Architecture

```
        training_data/                  eval/ground_truth.jsonl
              │                                   │
              ▼                                   │
      ┌───────────────┐                           │
      │  Data loader  │  [deterministic]          │
      │  + formatter  │                           │
      └───────┬───────┘                           │
              │                                    │
              ▼                                    │
      ┌───────────────────┐                        │
      │  LoRA / QLoRA      │                       │
      │  fine-tuning        │                      │
      │  (Unsloth backend)  │                      │
      └───────┬─────────────┘                      │
              │                                     │
     checkpoint-N saved                             │
              │                                     │
              ▼                                     ▼
      ┌─────────────────────────────────────────────────┐
      │             Evaluation harness                   │
      │  - task metrics (accuracy, F1, BLEU/ROUGE, etc.) │
      │  - factual-reliability suite:                    │
      │      confabulation rate, factual accuracy,       │
      │      abstention rate ("I don't know" when unsure)│
      └───────────────────┬───────────────────────────────┘
                          │
                          ▼
              per-checkpoint metrics.json
                          │
                          ▼
              training-run report (Markdown / HTML)
```

Fine-tuning and evaluation are decoupled: any checkpoint can be re-evaluated independently, and the eval suite runs identically whether the checkpoint came from slm-forge or another pipeline.

---

## Models

| Role | Examples | Why |
|---|---|---|
| Base SLMs supported | Phi-4-mini, Mistral-7B, Qwen2.5-7B, SmolLM2-1.7B | Range of sizes for different compute budgets |
| Fine-tuning backend | Unsloth (LoRA/QLoRA) | Fast, memory-efficient, single-GPU friendly |
| Teacher models (optional, distillation only) | Claude, GPT (API) | Used only to generate distillation data — never deployed as the served model |

All fine-tuned models run and are evaluated locally. Teacher-model API calls are optional and only used during data preparation, not at inference or eval time.

---

## Requirements

- Python 3.11+
- A CUDA-capable GPU (or Apple Silicon via Unsloth's MPS support) — 16 GB+ VRAM/unified memory recommended for 7B-class models with QLoRA
- [Unsloth](https://github.com/unslothai/unsloth) and its dependencies

---

## Installation

```bash
git clone https://github.com/TAJANIYoussef/slm-forge.git
cd slm-forge

uv pip install -e ".[dev]"
# or: pip install -e ".[dev]"
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

The factual-reliability suite scores three things per checkpoint:

| Metric | What it measures |
|---|---|
| Factual accuracy | Rate at which the model's claims match ground truth |
| Confabulation rate | Rate at which the model confidently states something false |
| Abstention rate | Rate at which the model correctly declines to answer when it lacks grounding |

### Labeling format

Add entries to `data/eval/ground_truth.jsonl` (one JSON object per line):

```json
{
  "prompt": "What is the maximum LTV ratio for a Mécanisme 5 housing loan?",
  "expected_answer": "70%",
  "language": "fr",
  "domain": "finance",
  "allow_abstention": true
}
```

Fields: `prompt` (required), `expected_answer` (required), `language` (required — `darija`, `fr`, or `en`), `domain` (optional), `allow_abstention` (optional, defaults to `true`).

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

- **7B-class models with QLoRA need ~16 GB VRAM/unified memory**; smaller machines will need smaller base models or slower training.
- **Ground-truth eval sets must be hand-labeled** per domain and language — the suite doesn't auto-generate them.
- **Darija support depends on base-model tokenizer coverage**; some SLMs handle Darija (especially Arabizi/Latin-script Darija) better than others.
- **Distillation from teacher models is optional and one-time** — it is not required to use the fine-tuning or evaluation pipeline.

---

## License

MIT