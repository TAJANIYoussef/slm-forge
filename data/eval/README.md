# Evaluation data

| File | What it is |
|---|---|
| `ground_truth.sample.jsonl` | 24 **fictional** items used by tests, docs and `configs/tiny-cpu-test.yaml`. Committed. |
| `ground_truth.jsonl` | Your real ground-truth set (finance/legal facts with sources). The example configs point here. |
| `labels/responses_gold.jsonl` | Human-labelled model responses for `slm-forge validate-detectors` (Track D). |

Every file uses the format in [`docs/dataset.md`](../../docs/dataset.md): UTF-8 JSONL, one item
per line; blank lines and lines starting with `//` are ignored. Validate a file before using it:

```bash
slm-forge data validate data/eval/ground_truth.sample.jsonl
slm-forge data validate data/eval/ground_truth.jsonl --strict   # warnings become errors
```

## The fictional sample

Everything in `ground_truth.sample.jsonl` is invented: the bank (*Banque Atlas Fictive*), its
products (*Prêt Atlas Habitat*, *Compte Épargne Zénith*, *Carte Atlas Platine Infinie*) and the
law (*Law No. 77-19 on Atlas digital cooperatives*). Tests and docs must never depend on
real-world facts, so they only ever use this file or fixtures like it.

| fact_id | Type | Domain | Buckets | Answer |
|---|---|---|---|---|
| `fin-atlas-ltv` | percent (`allow_abstention: false`) | finance | en, fr, darija_arab, darija_latn | 70% |
| `fin-zenith-deposit` | currency | finance | en, fr, darija_arab, darija_latn | 500 MAD |
| `fin-zenith-launch` | date | finance | en, darija_arab | 2019-05-12 |
| `leg-coop-article` | citation | legal | en, fr, darija_arab, darija_latn | Article 14 of Law 77-19 |
| `leg-coop-esign` | boolean | legal | fr, darija_latn | yes |
| `leg-coop-founders` | numeric (`split: dev`) | legal | fr, darija_arab | 5 |
| `fin-atlas-platine-u` | unanswerable: `nonexistent_entity` | finance | en, fr, darija_arab, darija_latn | null |
| `leg-coop-abolish-u` | unanswerable: `false_premise` | legal | en, darija_latn | null |

That gives 6 items per core bucket, 6 of 24 unanswerable (25%), and both domains in every
bucket. Translations of one fact share a `fact_id`, which is the cluster key for the bootstrap,
the subsample and cross-lingual consistency, so they agree on `answerable`, `domain`,
`answer_type` and the typed value. Darija items carry aliases in both scripts (Arabic and
Arabizi), and numeric items carry `known_false` distractors.

## Writing real items

- `expected_answer: null` means unanswerable and needs `unanswerable_reason`. The key itself
  is required, so a forgotten answer can never silently become an unanswerable item.
- Give typed answers (`answer_type` plus `numeric` / `date` / `citation`) whenever you can.
  They are matched with tolerances instead of string containment.
- Add `aliases` in every script a model might answer in, and plausible wrong values as
  `known_false`.
- Finance and legal facts go stale: set `source.as_of`, and re-check items older than 18 months.
- Keep a whole `fact_id` in one split. Only `dev` may be used to tune lexicons, τ or weights.
- Never add teacher-model (distilled) output: `assert_not_distilled` rejects records with
  `provenance.kind: distilled`.
