# Architecture

How slm-forge is put together: modules, the import rule, the data flow from ground truth to
report, the run directory, training backends, reproducibility and exit codes. Metric definitions
are in `docs/metrics.md`; the dataset format is in `docs/dataset.md`.

## 1. Module tree

Milestone column: **M0** = implemented now; otherwise the ROADMAP milestone(s) that implement the
remaining stubs (`grep -rn 'todo("M<k>"' src/` lists them).

```
src/slmforge/
├── __init__.py                 __version__                                                        M0
├── _stub.py                    todo(milestone, what) -> NotImplementedYet                          M0
├── cli.py                      typer app; lazy imports; exception -> exit code mapping              M0
├── types.py                    shared Literals and constants (buckets, answer types, reasons)      M0
├── repro.py                    canonical hashing, provenance, seeding, RNG preservation, offline   M0
├── protocol.py                 Protocol (everything that changes a score), lock, drift             M0 (lock); build: M2
├── config/
│   ├── schema.py               pydantic models: RootConfig, EvalConfig, DistillConfig, ...         M0
│   └── load.py                 YAML loading, --set dotted overrides, ${VAR} expansion, dump        M0
├── text/                       torch-free multilingual text layer
│   ├── normalize.py            L0/L1 normalization with offset map, tokens, script profile         M1; clitics: M4
│   ├── arabizi.py              Arabizi digit roles, cross-script consonant skeleton (L2)           M4
│   ├── numbers.py              quantities, units, multipliers, tolerance check (within)            M1 (en/fr); M4 (Darija/MSA)
│   ├── dates.py                partial dates incl. Moroccan/Arabizi month names                    M4
│   ├── citations.py            article/instrument references (article 12, الفصل 12, loi 31-08)     M4
│   ├── langid.py               rule-based Darija/MSA/fr/en identification; optional GlotLID        M4; GlotLID: M8
│   └── lexicons.py             versioned cue lexicons: load/validate/hash (M0), find (M1)          M1
├── resources/                  package data, versioned and hashed into the protocol
│   ├── lexicons/v1/{abstain,premise,hedge,filler,contrast,stopwords,yes_no}/{en,fr,msa,darija_arab,darija_latn}.yaml
│   ├── numbers/{en,fr,darija_arab,darija_latn}.yaml   units.yaml   months.yaml
│   └── prompts/default_v1/{en,fr,darija_arab,darija_latn}.yaml
├── data/
│   ├── loader.py               training records, formats, language resampling, assistant masks     M3
│   ├── contamination.py        train/eval overlap guard (M3), TrainIndex + fact_seen (M8)          M3, M8
│   └── distill.py              optional teacher distillation: gates + PII redaction (M0), run (M9) M9
├── train/
│   ├── lora.py                 finetune() orchestration, run/adapter manifests                     M3
│   ├── targets.py              LoRA target modules per architecture                                M0
│   ├── callbacks.py            posthoc markers (M3), inline eval callback (M6)                     M3, M6
│   ├── worker.py               `python -m slmforge.train.worker`: Unsloth training subprocess      M6
│   └── backends/
│       ├── __init__.py         TrainBackend protocol, select_backend (decided before any HF import) M0
│       ├── hf_peft.py          AutoModelForCausalLM + peft; CPU, MPS, CUDA                         M3; QLoRA: M6
│       └── unsloth.py          FastLanguageModel loading + LoRA (NVIDIA CUDA as tested)           M6
├── eval/
│   ├── dataset.py              ground-truth schema, loading, validation (M0); subsample (M3)       M0, M3, M5
│   ├── prompts.py              per-locale templates (M0), chat-template rendering (M2)             M2
│   ├── checkpoints.py          CheckpointRef, discovery, selection (M0); fingerprints (M2)         M2
│   ├── generate.py             torch-free generation contract: records, Generator, Fake/Replay     M0; cache: M2
│   ├── generate_hf.py          HFGenerator (+ teacher-forced logprobs), adapter hot-swap           M2, M3; inline: M6
│   ├── generate_vllm.py        optional VLLMGenerator (not bitwise deterministic)                  M9
│   ├── extract.py              clean_response, answer region (M0); candidates (M1)                 M1
│   ├── abstention.py           SignalDetector: abstain / premise / hedge signals                   M1
│   ├── match.py                matching cascade S1-S5 + false denial                               M1
│   ├── confidence.py           answer-span confidence from stored logprobs                         M1; consistency: M9
│   ├── outcomes.py             Outcome, Evidence, decide/reclassify (M0); classify, detect_invalid M1
│   ├── factual_reliability.py  metric table, bounds, macro, tau sweep (M0); suite scoring (M1)     M1, M4, M5
│   ├── calibration.py          ECE (equal-mass), Brier, AUROC, AURC                                M5
│   ├── stats.py                Wilson, Holm, exact McNemar (M0); cluster bootstrap, paired (M5)    M5
│   ├── task_metrics.py         optional task suite: chrF++, ROUGE-L, EM, token F1, eval loss       M3, M8
│   ├── review.py               review queue export/import, VerdictCache (SQLite)                   M7
│   ├── judge.py                detector validation gates (M7); LocalHFJudge (M9)                   M7, M9
│   ├── suites.py               suite registry, resolved lazily                                     M0
│   └── runner.py               EvalRunner: eval / eval-run / score / rescore, metrics.json, flags  M2, M3, M5
├── diagnostics/
│   └── tokenizer.py            tokenizer fertility, byte fallback, Arabizi digit isolation         M8
└── report/
    ├── render.py               Markdown (M1/M3) and self-contained HTML (M5) reports, trend.csv    M1, M3, M5
    └── svg.py                  dependency-free inline SVG charts                                   M5
```

Repository root: `configs/` (model configs, `tiny-cpu-test.yaml`, `eval/factual-reliability.yaml`,
`distill.yaml`, `protocols/`), `data/eval/` (fictional sample, your ground truth, labels),
`data/train/`, `tests/{unit/core,unit,integration,fixtures}/`, `docs/`.

## 2. Import rule (torch-free core)

| Rule | Detail |
|---|---|
| Core install | `pip install slm-forge` = pydantic, typer, pyyaml, numpy. Enough for `data validate\|stats`, `score`, `rescore`, `report` |
| Heavy modules | torch, transformers, peft, unsloth, bitsandbytes, vllm, scipy, sklearn |
| Who may import them | only `eval/generate_hf.py`, `eval/generate_vllm.py` (vllm), `eval/judge.py` (inside `LocalHFJudge`), `train/backends/hf_peft.py`, `train/backends/unsloth.py`, `train/callbacks.py` (inside the factory function) |
| How | **only inside functions** (lazy). Heavy types appear only under `TYPE_CHECKING` |
| Invariant | importing **any** `slmforge` module, and `slm-forge --help`, loads none of the heavy modules. Enforced by `tests/unit/core/test_import_rule.py` in a subprocess |
| Unsloth | never imported in the CLI process or in pytest; only as the first import of the training subprocess (`train/worker.py`) |
| Backend choice | `train.backends.select_backend` decides from `importlib.util.find_spec` before transformers is imported, because Unsloth patches transformers/peft/trl on import |
| CUDA env | `cli.main` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before any torch import |
| SciPy / scikit-learn | test oracles only (dev extra) |

## 3. Data flow

```
 data/eval/ground_truth.jsonl ──► dataset.load_ground_truth ──► GroundTruthSet (validated, sha256)
                                          │ filter(split, reviewed) ─► subsample(fraction, seed)  ─► subsample_ids.json
                                          ▼
 configs/*.yaml ──► config.load ──► EvalConfig ──► protocol.build_protocol ──► protocol.json  (+ optional lock check)
                                          │
                                          ▼
 prompts.PromptBuilder (template default_v1 + chat template) ──► GenRequest(item_id, content_hash, lang_script, messages)
                                          │
                                          ▼
 Generator (HFGenerator | VLLMGenerator | ReplayGenerator | FakeGenerator)
   ├─ greedy decode with a fresh GenerationConfig, length-sorted batches per lang_script budget
   └─ teacher-forced pass over prompt + response ─► token_logprobs, token_char_spans
                                          │   GenerationCache (key = checkpoint fp, generator fp,
                                          │   decoding sha, prompt sha, content hash)
                                          ▼
 generations.jsonl   one GenerationRecord per item  ◄── or any external {"item_id", "response"} JSONL (slm-forge score)
                                          │
                                          ▼
 scoring pipeline (torch-free, FactualReliabilitySuite.score_item)
   T0 clean_response ─ T1 detect_invalid ─ T2 find_answer_region ─ T3 SignalDetector.detect
   T4 extract_candidates ─ T5 match ─ T6 compute_confidence ─ T7 classify/decide ─ T8 verdict override
                                          │                                   ▲
                                          ▼                                   │ VerdictCache (human), judge
 scored.jsonl        one ItemResult per item (evidence, outcome, best/worst case, rule)
                                          │
                                          ▼
 FactualReliabilitySuite.summarize ─► metrics.json   (micro/macro/bucket rates, bounds, CIs, calibration,
                                          │            tau sweep, flags, provenance, four identity hashes)
                                          ▼  (eval-run: every scheduled checkpoint, same subsample)
 runner.compare_checkpoints + compute_flags ─► run_metrics.json, trend.csv, review_queue.jsonl
                                          │
                                          ▼
 report.render ─► report.md, report.html (inline SVG)
```

Two contracts carry everything:

| Contract | Type | File | Minimal content |
|---|---|---|---|
| generation → scoring | `eval.generate.GenerationRecord` | `generations.jsonl` | `{"item_id", "response"}`; optional `finish_reason`, `content_hash`, `prompt_sha256`, `token_logprobs`, `token_char_spans`, `n_prompt_tokens`, `n_new_tokens`, `latency_s` |
| scoring → statistics | `eval.outcomes.ItemResult` | `scored.jsonl` | item keys, `evidence`, `outcome`, `best_case`, `worst_case`, `hedged`, `decision_rule`, `verdict_source`, `confidence_source`, fidelity fields, `response_sha256`, `flags` |

`rescore` reads `generations.jsonl` and never regenerates: lexicon edits, τ changes and human
verdicts are applied on CPU.

### 3.1 Generation cache key

`cache_key(checkpoint_fp, generator_fp, decoding_sha256, prompt_sha256, content_hash)` (now) =
sha256 of the five parts. Never a path: `checkpoint-500` exists in every run.

| Part | Covers |
|---|---|
| `checkpoint_fp` | adapter: `adapter_config.json` + weights hash, plus base model id and resolved revision; full model: `config.json` + safetensors headers |
| `generator_fp` | backend, model identity, dtype, quantization, attention implementation, batch size (GPU), device, torch/transformers versions |
| `decoding_sha256` | `DecodingParams.sha256()`: per-bucket `max_new_tokens`, stop strings, seed, greedy, `repetition_penalty = 1.0` |
| `prompt_sha256` | the rendered prompt (template + chat template) |
| `content_hash` | `GroundTruthItem.content_hash` (scoring-relevant item fields) |

The cache is append-only JSONL, fsync'd per batch, so a killed run resumes where it stopped (M2).

## 4. Run directory layout

```
runs/<run.name>/
├── run_manifest.json          resolved config + config_sha256, provenance, base model + resolved revision,
│                              backend, data fingerprints, contamination summary, status (running|finished)
├── resolved_config.yaml       config after --set overrides (also the Unsloth worker's input)
├── train_log.jsonl            trainer log history: step, loss, eval_loss, lr, grad_norm
├── checkpoint-<step>/         untouched HF/PEFT directory
│   ├── adapter_config.json, adapter weights, tokenizer files
│   ├── slmforge_adapter_manifest.json   base repo + revision, architecture, target modules,
│   │                                    vocab size, tokenizer hash, quantization
│   └── .eval_pending | .eval_final       posthoc markers written by the callback
└── eval/<suite>/<protocol8>/
    ├── protocol.json   subsample_ids.json   eval_config.yaml   contamination.json
    ├── base/                  generations.jsonl  scored.jsonl  metrics.json
    ├── checkpoint-<step>/     generations.jsonl  scored.jsonl  metrics.json
    ├── run_metrics.json   trend.csv   review_queue.jsonl
    ├── report.md
    └── report.html
runs/_shared/verdicts.sqlite                     human/judge verdict cache, shared across runs
eval_out/<ckpt-slug>/<suite>/<protocol8>/...     standalone `slm-forge eval` output
```

- `<protocol8>` = first 8 hex chars of the protocol hash: different protocols never mix in one trend or report.
- Standalone `eval` never writes into the checkpoint directory or the HF cache.
- A fake generator never writes into a run directory (one that has `run_manifest.json`).
- `eval_config.yaml` is what `slm-forge rescore DIR` reloads.

### 4.1 Eval schedule and cost

| Setting | Default | Meaning |
|---|---|---|
| `eval.schedule.mode` | `posthoc` | the callback only marks checkpoints; `eval-run` evaluates them afterwards in a separate step, never concurrently on the same GPU |
| `eval.schedule.mode: inline` | off | M6: evaluate in `on_save` through the same `HFGenerator` path, inside a context that sets `eval()`, `inference_mode`, `use_cache=True`, left padding, no autocast, Unsloth inference/training toggles, and restores every RNG stream. Inline == posthoc is claimed only on CPU fp32 |
| `every_n_steps` | None | must be a multiple of `train.save_steps` (validated) |
| `include_base`, `include_final` | true | base = step 0, adapters disabled |
| `subsample.fraction` | None (full set) | fixed cluster-level subsample, identical at every checkpoint **including base and final**; `full_at: [base, final]` adds separate full-set endpoint tables |
| one base load | — | `eval-run` loads the base once and hot-swaps adapters |
| `--dry-run` | — | prints the cost model: `n_gen = sum over checkpoints of items`, token upper bound `sum_i max_new_tokens[bucket_i]`, seconds from a cached throughput probe |

## 5. Training backends

| Backend | Runs on | Process | Notes |
|---|---|---|---|
| `unsloth` | NVIDIA CUDA (as tested) | `python -m slmforge.train.worker` subprocess whose **first** import is `unsloth`; the parent reads `run_manifest.json` and checkpoints afterwards | pinned in `requirements/cuda-unsloth.lock`; the CPU and CUDA locks pin the same transformers/peft/trl versions (M6) |
| `hf_peft` | CPU (CI), MPS (LoRA, best effort), CUDA | in process | reference backend; plain `transformers.Trainer` with an assistant-mask collator (M3); QLoRA via bitsandbytes on CUDA (M6) |
| `auto` | — | — | `unsloth` iff it is importable **and** CUDA is available, else `hf_peft` |

Rules (each failure is `BackendUnavailable`, an `OSError`, exit code 3):

- QLoRA (`quant.load_in_4bit`) needs bitsandbytes and CUDA. It never silently downgrades to 16-bit LoRA; on CPU it is an error.
- Labels come from **assistant masks** (turn boundaries), never from the pad id: base Qwen2.5 sets pad == eos and Mistral has no pad, so pad-id masking would stop EOS from being trained. `assert_final_eos_labeled` checks that every example's final EOS has a real label.
- Every checkpoint stores its tokenizer and `slmforge_adapter_manifest.json`. Before an adapter is loaded on a base, modules and vocab are checked (`AdapterIncompatible`, exit 3). Vocabulary resizing is off by default; a 4-bit base is never merged.
- `finetune` order: select backend → run dir + `resolved_config.yaml` + `run_manifest.json` (`running`) → train (subprocess for Unsloth) → free GPU, manifest `finished` → posthoc `eval-run` if `eval.schedule.after_training` and not `--skip-eval`.

## 6. Reproducibility

| Concern | Mechanism |
|---|---|
| Seeds | `run.seed` seeds random, numpy, torch, CUDA and `transformers.set_seed`; `data_seed` is recorded separately (defaults to `seed`) |
| Determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch; `repro.set_determinism`: `torch.use_deterministic_algorithms(True, warn_only=not run.strict_determinism)`, cuDNN deterministic, benchmark off; inline eval inside `repro.preserved_rng_state()` |
| Decoding | greedy, `num_beams=1`, from a **fresh** `GenerationConfig` (never the model's shipped one): `do_sample=False`, `repetition_penalty=1.0`, `no_repeat_ngram_size=0`, temperature/top_p/top_k unset, explicit EOS, end-of-turn and pad ids. The effective config is written to `metrics.json`. `attn_implementation` is set explicitly (eager and sdpa differ numerically) |
| Confidence | teacher-forced logprobs, identical for HF, Unsloth, vLLM and replayed responses scored later |
| Identity | `config_sha256` (canonical JSON minus `NON_SEMANTIC_FIELDS`, after `--set`), `Protocol.sha256`, dataset sha256, generator fingerprint: the four hashes in every `metrics.json` |
| Pins | model `revision` pinned to a commit SHA (branch names warn) or resolved and recorded; tokenizer and chat-template hashes recorded; dependency locks `uv.lock` and `requirements/cuda-unsloth.lock` (M6) are to be committed (not in the tree yet) |
| Provenance | slm-forge version, git commit and dirty flag, Python, tracked package versions, CUDA/cuDNN/driver/GPU, torch thread count, determinism flags, offline mode, timestamp |
| Offline | `repro.offline_mode()` sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE` before HF imports; a missing local model is exit 3. Eval makes no network calls |
| Bootstrap RNG | indices from `PCG64(seed).random_raw()` with an own reduction (stable across NumPy versions) |
| Golden files | compared through a canonical projection: no timings or environment, floats rounded to 1e-9 |

Guarantees:

| Setting | Guarantee |
|---|---|
| CPU, fp32 | bit-identical `generations.jsonl` and `metrics.json` across runs and across batch sizes 1 and 4, apart from timestamps and timing; resumed run == uninterrupted run |
| GPU | best effort on the same hardware, driver and config; GPU backward passes (Triton, flash-attention) are not bitwise reproducible |
| vLLM | labelled non-bitwise-deterministic |
| `rescore` | always reproducible: scoring is a pure function of `generations.jsonl`, the ground truth and the protocol |

## 7. Exit codes

| Code | Meaning | Raised by |
|---|---|---|
| 0 | ok | — |
| 1 | data, config or protocol validation error (incl. `ProtocolDrift`, `GroundTruthError`, pydantic `ValidationError`, missing input file) | `data validate`, config loading, protocol lock check |
| 2 | usage error (bad flags or arguments) | typer |
| 3 | environment: CUDA or 4-bit unavailable, backend not installed, model not available offline, incompatible adapter (`OSError` subclasses such as `BackendUnavailable`, `AdapterIncompatible`) | backends, checkpoint loading |
| 4 | a quality gate failed (`validate-detectors` with any component below its gate) | `validate-detectors` (gates computed in M7) |
| 5 | not implemented yet (`NotImplementedYet`; the message names the milestone) | any stub |
