# slm-forge roadmap

This is the development plan from the current skeleton (M0) to a usable v0.1 and on to a v1.0 you can cite.
The detailed specs are in [`docs/metrics.md`](docs/metrics.md), [`docs/dataset.md`](docs/dataset.md) and
[`docs/architecture.md`](docs/architecture.md). Each unimplemented function in `src/` raises
`NotImplementedYet("M<k>")`, naming the milestone that delivers it:

```bash
grep -rn 'todo("M1"' src/        # everything M1 still has to implement
slm-forge score ...              # a stub command exits with code 5 and names its milestone
```

---

## Goal

**Measure when fine-tuning makes a small model less reliable.** At each evaluated checkpoint, slm-forge reports:

- **factual accuracy**
- **confabulation rate**: the model is wrong, did not hedge, and was confident
- **abstention rate**: the model correctly declines on unanswerable items

These are reported per language and script bucket (Darija in Arabic script, Darija in Arabizi, French, English), with confidence intervals. slm-forge also flags the step at which reliability starts to degrade while training loss keeps improving.

## Principles

1. **Eval is decoupled from training.**
   - It can score any HF checkpoint, any PEFT adapter, or a plain `{item_id, response}` JSONL from any pipeline.
   - Generation and scoring communicate only through `generations.jsonl`.
   - `rescore` never regenerates.
2. **The core is torch-free.** Scoring, validation and reports install with `pip install slm-forge` (pydantic, typer, pyyaml, numpy). The ML stack is an extra. A test enforces this.
3. **Every number is defined precisely.**
   - Each item gets exactly one outcome, from a pure, versioned decision table.
   - Every rate has an explicit denominator, and INVALID and UNRESOLVED items are never dropped from it.
   - Items the rules cannot decide are reported as bounds, never silently guessed.
4. **Darija is a first-class target.**
   - Lexicons cover both Darija scripts and code-switching.
   - Moroccan number formats are parsed.
   - The headline metric is a macro average over language/script buckets, so French and English volume cannot hide a Darija collapse.
5. **Offline and reproducible.**
   - Eval makes no network calls.
   - Each result is keyed by a protocol hash: scorer, lexicon, prompt and dataset versions, plus τ and the subsample.
   - On CPU fp32, outputs are bit-identical across runs. On GPU this is best effort.

## Decisions that answer the README review

| Open question | Decision | Where |
|---|---|---|
| How is an answer judged to match? | Typed cascade, in order: exact match; typed numeric/date/citation/boolean/choice with tolerances; whole-token alias containment; Darija consonant skeleton. `known_false` distractors give a decisive mismatch. Anything undecided is UNRESOLVED, handled with bounds and human review. No LLM judge by default | `eval/match.py`, docs/metrics.md |
| "Confidently false" vs merely wrong | CONFABULATION = wrong, **no verbal hedge**, **and** answer-span confidence ≥ τ. Confidence comes from a teacher-forced pass. WRONG_LOW_CONF and WRONG_HEDGED are separate outcomes. The τ-free `unhedged_error_rate` is always reported, because fine-tuning sharpens confidence | `eval/outcomes.py` (implemented) |
| Abstention across Darija / FR / EN | Versioned lexicons for 5 buckets (en, fr, msa, darija_arab, darija_latn). **All buckets apply to every response**, which handles code-switching. Strong vs weak cues: courtesy formulas never count as a hedge. abstained = cue AND no committed answer | `text/lexicons.py`, `resources/lexicons/v1/` |
| Unanswerable items | `expected_answer: null` + `unanswerable_reason`. The key is required, so a forgotten field cannot silently make an item unanswerable | `eval/dataset.py` (implemented) |
| `allow_abstention` semantics | Applies to answerable items only. `true` means abstaining is neutral (utility 0). `false` means it is required knowledge (ABSTAIN_UNWARRANTED, −0.5). Abstaining is never "correct" for accuracy | `eval/outcomes.py` (implemented) |
| Reproducibility | Seeds, a fresh greedy `GenerationConfig` (shipped sampling/repetition defaults never leak in), pinned model revisions, `uv.lock`, canonical config hashes, the protocol hash, and provenance in every `metrics.json` | `repro.py`, `protocol.py` |
| Eval cost at every checkpoint | Eval every N steps on a **fixed cluster-level subsample, identical at every checkpoint** (base and final included). The full set at base/final goes into separate endpoint tables. The base model is loaded once and adapters are hot-swapped. A generation cache plus `--dry-run` gives a cost estimate | `eval/runner.py` |
| Unsloth / Apple Silicon | Unsloth is treated as **NVIDIA CUDA only** and runs in a training subprocess. A plain HF + PEFT backend runs everywhere, including CPU (CI) and MPS for LoRA | `train/backends/` |
| Distillation ToS | Off by default. Two explicit opt-ins (`tos_acknowledged` + date, `allow_send_data`), PII redaction and provenance. Teacher output never enters eval sets | `data/distill.py`, docs/distillation-terms.md |

---

## Milestones

Estimates assume one developer working part-time. Track D (data) runs in parallel from the start, and it is the real critical path.

```
M0 skeleton ──► M1 offline scoring ──► M2 generation/eval ──► M3 finetune + eval-run ══► v0.1
                       │                                              │
                       └──► M4 Darija text layer ──► M5 statistics ◄──┘ ══► v0.2
                                                         │
              M6 Unsloth/QLoRA ── M7 review + validation gates ── M8 contamination/diagnostics ══► v1.0
                                                         │
                                                   M9 optional extras
Track D (ground truth + gold labels) ═══════════════════════════════════════════════════════════════►
```

### M0: Skeleton and tooling ✅ (this commit)

- `pyproject.toml` (hatchling, src layout, torch-free core, extras `hf`, `qlora`, `unsloth`, `darija`, `distill`, `dev`), ruff, mypy, CI (core-minimal + full-cpu).
- Every module in `docs/architecture.md` exists with typed signatures and docstrings. Unimplemented bodies raise `NotImplementedYet("M<k>")`.
- **Implemented now:** shared types; config schema and loader (`--set` overrides, `${VAR}` expansion); the ground-truth schema with per-item and cross-item validation; the outcome decision table, `reclassify` and the τ sweep; the declarative metric table with UNRESOLVED bounds and macro averaging; L0 normalization and script profiles; lexicon and unit-table loading with L1-form validation; final-answer region detection; Wilson, Holm and exact McNemar; the generation record contract with fake and replay generators; provenance and hashing; the protocol model; prompt templates; checkpoint discovery; backend selection; PII redaction; distillation gates; the CLI (`data validate` works).
- Example configs for Qwen2.5-7B, Phi-4-mini, Mistral-7B and SmolLM2-1.7B, plus a tiny CPU test config. A 24-item **fictional** sample ground-truth set covering 4 buckets.
- **Exit criteria:**
  - `slm-forge --help` lists every command.
  - `slm-forge data validate data/eval/ground_truth.sample.jsonl` passes.
  - Importing any module loads no torch.
  - The test suite is green.

### M1: Offline scoring vertical slice (~2 weeks)

*After M1, you can score responses from any model, as long as they are in French or English.*
- `text/normalize.py`: L1 matching normalization with an offset map (NFKC, digit folding, diacritics, Arabic letter variants, punctuation), plus tokenization.
- `text/numbers.py`: en/fr numbers, percents and money (`70 %`, `1 000,50 DH`, `1,000.50`, scale words), with locale ambiguity flags.
- `text/lexicons.py` `find()` plus seed lexicons for en and fr: abstain, premise, hedge, filler, contrast.
- `eval/extract.py`, `eval/match.py` (stages S1–S3, S5, the distinct-values rule, false denial), `eval/abstention.py`, `eval/confidence.py` (from stored logprobs), `outcomes.classify`, `detect_invalid`.
- `FactualReliabilitySuite.score`; `slm-forge score` and `rescore`; a Markdown report for one checkpoint.
- `tests/fixtures/scoring_cases.jsonl`: ≥ 150 hand-labelled (response → outcome) cases in en/fr, including the adversarial ones ("I don't know why people say 60%; it's 70%", "70 %, pas 80 %").
- **Exit criteria:**
  - Outcome agreement with the fixtures ≥ 95%.
  - Fuzzing with random Unicode never raises.
  - Golden `metrics.json` is reproduced byte-for-byte, through a canonical projection that leaves out timings.

### M2: Generation and single-checkpoint `eval` (~1.5 weeks)

- `eval/prompts.py`: chat-template rendering, with a fallback that folds the system prompt into the user turn (Mistral).
- `eval/checkpoints.py`: fingerprints and an adapter manifest with a base-compatibility check.
- `eval/generate_hf.py`:
  - left-padded, length-sorted batches grouped by bucket;
  - a **fresh** greedy `GenerationConfig`;
  - **teacher-forced** token logprobs with character spans.
- `GenerationCache` (content-hash keys, resumable); `protocol.build_protocol`; the `metrics.json` writer and JSON Schema; `slm-forge eval`.
- **Exit criteria, on a tiny model on CPU in fp32:**
  - Logprobs match a manual forward pass within 1e-4.
  - batch_size 1 and 4 give identical outputs.
  - Two runs give identical outputs.
  - A run killed and resumed equals an uninterrupted run.
  - The no-network guard holds.

### M3: Fine-tuning and `eval-run` → **v0.1.0** (~2 weeks)

- `data/loader.py`:
  - chat, prompt_response and alpaca formats;
  - **assistant-mask labels** built from turn boundaries, never from pad id (Qwen's pad == eos would otherwise stop EOS from being trained);
  - language resampling (`alpha`) and the abstention mix.
- The HF-PEFT backend, `finetune()`, `run_manifest.json`, and the posthoc callback (marks checkpoints for eval).
- `EvalRunner.eval_run`: the base model loaded once, adapters hot-swapped, a fixed subsample, and a trend table and charts in Markdown.
- The minimal contamination guard: exact or 8-gram overlap with eval prompts aborts training.
- **Exit criteria:**
  - On CPU, 6 steps with save_steps 2 produce 3 checkpoints, then an eval-run and a report.
  - On 8 fictional Q/A pairs, factual_accuracy rises by ≥ 0.5 from base to final.
  - The final EOS of every example has a real label (a tested invariant).
  - Re-running eval-run regenerates nothing.

### M4: Darija text layer (~2 weeks; can start in parallel after M1)

- Arabizi:
  - digit roles: `3la`, `s7i7` and `9ism` are letters; `2eme`, `70dh` and `3ans` are numerals;
  - a shared Arabic/Latin consonant skeleton (`ma3reftch` ≈ `ما عرفتش`);
  - clitic stripping only when ≥ 3 letters remain.
- Darija and MSA lexicons: abstention, hedges (`kandon`, `كنظن`, `wa9ila`), premise rejection, yes/no.
- Moroccan quantities:
  - `sb3in f lmya`;
  - `ryal`/`centime` conversions;
  - the two readings of `melyoun` (1,000,000 vs 10,000 DH);
  - Eastern Arabic digits and `٫`/`٬`.
- Dates (Moroccan month names: `غشت`, `شتنبر`, `chtanbr`); legal citations (`الفصل 12`, `loi 31-08`, `l-fasl 12`); rule-based langid; language and script fidelity.
- Macro headline over the core buckets (`min_bucket_n = 30`), the Darija gap, the script gap, and cross-lingual consistency over `fact_id`.
- **Exit criteria:**
  - ≥ 300 table-driven text cases (60+ per Darija script, 30+ code-switched).
  - Hypothesis property tests: normalization is idempotent and the offset map is monotone.
  - Scoring fixtures reach ≥ 400 cases, with ≥ 90% agreement per core bucket.

### M5: Statistics and flags → **v0.2.0** (~1.5 weeks)

- Stratified **cluster bootstrap** over `fact_id`, because translations of one fact are correlated. Use BCa intervals and raise `CI_FEW_CLUSTERS` below 30 clusters.
- Paired comparison of each checkpoint against the base model, with Holm correction for each primary metric.
- `degradation_onset_step`, which must agree under both UNRESOLVED bounds.
- Flags: REGRESSION, TRADEOFF ("loss fell but reliability worsened"), HEDGE_INFLATION, OVER_ABSTENTION, DEGENERATION, LANGUAGE_DRIFT, plus the flip matrix (CORRECT→CONFAB examples).
- Calibration (equal-mass ECE, Brier, AUROC, AURC); a self-contained HTML report with inline SVG and RTL-safe Arabic; `data stats` (design effect, MDE).
- **Exit criteria:**
  - Synthetic confabulations injected from step 300 are detected with onset = 300 and a TRADEOFF flag.
  - The false-onset rate over simulated null runs is ≤ 0.05 + 2 SE.
  - CI coverage is checked with ≥ 2000 replications.

### M6: Unsloth + QLoRA on CUDA, inline eval (~1 week)

- `UnslothBackend` in a training subprocess whose first import is `unsloth`; `requirements/cuda-unsloth.lock`.
- QLoRA via bitsandbytes. QLoRA on CPU is an error (exit 3), not a silent downgrade.
- Optional inline eval in `on_save`: eval mode, KV cache, left padding, Unsloth `for_inference()`/`for_training()` toggles, preserved RNG state. It is off by default for 7B models on 16 GB.
- **Exit criteria:**
  - SmolLM2-1.7B QLoRA, 200 steps on one GPU, produces 2 evaluated checkpoints.
  - Inline metrics equal posthoc metrics on CPU fp32.
  - The training loss sequence is unchanged with the callback on.

### M7: Human review loop and validated detectors (~1.5 weeks; needs Track D3)

- `review export`/`import` with a SQLite verdict cache keyed by (item, normalized response). Human verdicts override rule verdicts on rescore and narrow the bounds.
- `validate-detectors` gates for each stratum, computed on the non-exact and UNRESOLVED tail rather than on easy items:
  - abstention P ≥ 0.95 and R ≥ 0.90;
  - hedge P and R ≥ 0.85;
  - matcher agreement ≥ 0.97.
- Failing strata are flagged `DETECTOR_UNVALIDATED` and are never used silently.
- `protocol lock` is enforced for published test-split results (`ProtocolDrift` → exit 1).

### M8: Contamination, tokenizer diagnostics, task metrics (~1 week)

- `TrainIndex`: skeleton shingles for Darija and L1 shingles otherwise, with optional MinHash. `slm-forge data contamination` and the `fact_seen` stratum.
- `diagnose-tokenizer`: fertility, byte fallback, Arabizi digit isolation and the token premium over English, per model and bucket. It sets per-bucket `max_new_tokens`.
- chrF++ (the primary fluency metric for Darija), ROUGE-L with a Unicode tokenizer, EM and token F1.

### M9: Optional extras

- Teacher distillation (Anthropic/OpenAI clients, lazily imported, ToS-gated, provenance on every record).
- A local judge, used only on rule-UNRESOLVED items and only after passing κ gates.
- A vLLM generator, labelled non-bitwise-deterministic.
- Sampling-consistency confidence.

### Track D: Ground truth and gold labels (parallel; needs ≥ 2 fluent annotators with domain knowledge)

| Stage | When | Content |
|---|---|---|
| D1 | by M1 | 80 seed items = 4 buckets × 2 domains × (7 answerable + 3 unanswerable), `fact_id` shared across buckets; `docs/annotation.md` |
| D2 | by M3 | 480 items (~120 per bucket, 25% unanswerable across all reasons, ≥ 60% typed answers); every unanswerable item checked by a second person |
| D3 | by M7 | `responses_gold.jsonl`: ≥ 600 real responses from SmolLM2 and Qwen2.5 (base and fine-tuned), 20% double-labelled, Krippendorff's α reported |
| D4 | v1.0 | 1,200 items (~150 per bucket × domain); enough for about a ±5pp paired MDE per bucket |

Without D2, the CIs are too wide to support any claim. Without D3, the detectors are unvalidated. Both are prerequisites for v1.0, not extras.

---

## Risks

| Risk | Mitigation |
|---|---|
| Lexical detectors mislabel hedges and abstentions, especially in Darija | Strong/weak cue split, complementizer suppression ("I don't know *why*…"), validation gates per stratum, UNRESOLVED instead of guessing, human review loop |
| Fine-tuning sharpens confidence, so confabulation at a fixed τ rises even when errors don't | τ-free `unhedged_error_rate` as the trend metric; confab decomposed into error × unhedged × confident; τ fixed per protocol (optionally calibrated on the base dev split) |
| Models learn to over-hedge or over-abstain to game the metric | HEDGE_INFLATION and OVER_ABSTENTION flags; `reliability_utility` penalizes both; coverage is reported |
| Small eval sets give noisy trends | Cluster bootstrap, a paired-vs-base test with Holm, MDE in `data stats`, ≥ 3 training seeds required for claims about fine-tuning |
| Unsloth import side effects and hardware limits | Subprocess isolation, never imported by tests, an HF-PEFT fallback, a pinned lock file |
| pad == eos masking makes models stop terminating (looks like a reliability trend) | Labels from assistant masks, a tested EOS invariant, a `truncation_rate` series and the DEGENERATION flag |
| Real-world facts go stale (rates, laws) | `source.as_of` on every finance/legal item, a STALE_FACTS flag; tests use fictional facts only |
| Distillation terms and data protection | Explicit opt-ins, PII redaction, Law 09-08 note, provenance, never in eval data |

## Working on a milestone

1. `grep -rn 'todo("M<k>"' src/` lists the stubs to implement.
2. Write the tests first, from the spec in `docs/`.
3. Keep the import rule: `pytest tests/unit/core/test_import_rule.py`.
4. A change to a lexicon, matcher or normalizer must bump its version constant, which changes the protocol hash, so old and new results are never mixed.
