# Ground-truth dataset: reference

The eval set for the factual-reliability suite. Model: `slmforge.eval.dataset.GroundTruthItem`
(pydantic v2, `frozen=True`, `extra="forbid"`). Where this page and `eval/dataset.py` disagree, the
code wins and this page has a bug. How to write items: `docs/annotation.md`.

Status markers: **(now)** implemented in the current tree; **(M<k>)** delivered by that ROADMAP
milestone.

## 1. File format

- Path: `data/eval/ground_truth.jsonl` (the example configs point there); the committed fictional sample is `data/eval/ground_truth.sample.jsonl`.
- UTF-8, one JSON object per line. A UTF-8 BOM is tolerated. Blank lines and lines starting with `//` are ignored.
- Load and validate: `slm-forge data validate FILE [--strict]` (now). Any error → exit code 1. `--strict` turns warnings into errors.
- Every fact in examples, tests and docs is **fictional**. Real items need a `source`.

## 2. Fields

| Field | Type | Default | Rule |
|---|---|---|---|
| `schema_version` | `1` | `1` | A record with neither `schema_version` nor `id` is a legacy record (§6) |
| `id` | str | **required** | `^[A-Za-z0-9][A-Za-z0-9._:-]{2,63}$` (3–64 chars); unique in the file (E02) |
| `fact_id` | str \| null | = `id` | Cluster key: paraphrases and translations of one fact share it. Used by the cluster bootstrap, the subsample, cross-lingual metrics and the split rule |
| `prompt` | str | **required** | 1–4000 characters |
| `language` | `darija` \| `fr` \| `en` | **required** | — |
| `script` | `arab` \| `latn` \| `mixed` \| null | null | fr/en resolve to `latn` (`arab` is E07). Darija must declare it (E07) until M1 detects it from the prompt |
| `code_switch` | list of `darija` \| `msa` \| `fr` \| `en` | `[]` | Other languages present in the prompt; `code_switched = bool(code_switch)` |
| `domain` | str | `general` | Must be in `eval.allowed_domains` (default `finance`, `legal`, `general`) when loaded through an eval config (E01) |
| `expected_answer` | str \| null | **key required** | `null` ⇔ unanswerable. `""` or whitespace is E03. A forgotten key is E03, so it can never silently become an unanswerable item |
| `answerable` | bool \| null | null | Resolved to `expected_answer is not None`; an explicit value must agree (E04) |
| `answer_type` | see §2.2 \| null | null | On unanswerable items it describes what the question asks for, so a guessed value is detected as a typed answer. Inferred when null (M1): percent → currency → date → numeric (single number) → boolean → free_text |
| `aliases` | list[str] | `[]` | Accepted surface forms, in **every** script a model may answer in. No empty alias (E14) |
| `known_false` | list[str] | `[]` | Plausible distractors; any one present gives a decisive MISMATCH |
| `numeric` | NumericSpec \| null | null | Required for numeric, percent, currency, duration (E08) |
| `date` | DateSpec \| null | null | Required for `answer_type: date` (E09). Python attribute `date_` |
| `citation` | CitationSpec \| null | null | Required for `answer_type: citation` (E09) |
| `choices` | list[str] | `[]` | For `choice`: must contain `expected_answer` (E09) |
| `allow_abstention` | bool | `true` | Answerable items only (§3); must be `true` on unanswerable items (E05) |
| `unanswerable_reason` | see §2.2 \| null | null | Required iff unanswerable (E05, E06) |
| `premise_rejection_cues` | list[str] | `[]` | Unanswerable items only (E06): extra phrases that count as a correct premise rejection |
| `context` | str \| null | null | Optional grounding passage; marks the item `grounded` |
| `source` | SourceRef \| null | null | Provenance and `as_of` date |
| `anchor_terms` | list[str] | `[]` | Entities for the `fact_seen` contamination check and prompt-echo exclusion |
| `split` | `dev` \| `test` | `test` | Only `dev` may be used to tune lexicons, τ, weights or judge prompts. A fact_id lives in one split (E13) |
| `reviewed` | bool | `true` | Teacher-assisted paraphrases enter as `false`; `eval.require_reviewed` excludes them |
| `label_version` | int | `1` | Bump when the label of an existing item changes |
| `annotators` | list[str] | `[]` | Not hashed into `content_hash` |
| `tags` | list[str] | `[]` | `legacy` is added on upgrade. Not hashed |
| `notes` | str \| null | null | Not hashed |
| `meta` | object | `{}` | Free-form. Unknown top-level keys are an error (E01): put them here. Not hashed |

### 2.1 Nested specs (all `extra="forbid"`)

| Spec | Field | Type | Default | Rule |
|---|---|---|---|---|
| `numeric` | `value` | Decimal (JSON number or string) | required | Use a string for exact decimals (`"3.25"`) |
| | `unit` | str \| null | null | `%`, `MAD`, `EUR`, `USD`, `days`, `months`, `years` or null |
| | `abs_tol` | Decimal ≥ 0 | 0 | Match iff `\|a − g\| ≤ max(abs_tol, rel_tol × \|g\|)` |
| | `rel_tol` | Decimal ≥ 0 | 0 | |
| | `accept_units` | list[str] | `[]` | Extra units accepted for `numeric` items |
| | `accept_fraction_for_percent` | bool | false | Accept `0.7` for `70%` |
| `date` | `value` | `YYYY` \| `YYYY-MM` \| `YYYY-MM-DD` | required | `granularity` (year / month / day) is the finest component present |
| `citation` | `article` | str | required | `"14"`, `"12-1"` |
| | `instrument` | str \| null | null | Canonical `NN-YY` form (`"77-19"`) |
| | `instrument_aliases` | list[str] | `[]` | "loi 77-19", "القانون 77-19", "l9anoun 77-19" |
| `source` | `citation` | str | required | Human-readable reference |
| | `url` | str \| null | null | |
| | `as_of` | ISO date \| null | null | The date the fact was checked to be true |

### 2.2 Enumerations (`slmforge.types`)

| Name | Values |
|---|---|
| `AnswerType` | `numeric`, `percent`, `currency`, `duration`, `date`, `boolean`, `choice`, `citation`, `entity`, `free_text` |
| TYPED (matched with tolerances) | numeric, percent, currency, duration, date, boolean, choice, citation |
| `UnanswerableReason` | `false_premise`, `nonexistent_entity`, `future_or_unknowable`, `not_in_any_source`, `underspecified` |
| `LangScript` (bucket) | `darija_arab`, `darija_latn`, `darija_mixed`, `fr`, `en`; core buckets: darija_arab, darija_latn, fr, en |

### 2.3 Derived values

| Value | Definition |
|---|---|
| `fact_id`, `script`, `answerable` | resolved as in §2 once, at validation time |
| `lang_script` | `darija_<script>` for Darija, otherwise the language code |
| `code_switched` | `bool(code_switch)` |
| `grounded` | `context is not None` |
| `content_hash` | `sha256(canonical_json(item minus notes, tags, annotators, meta))[:16]`, field names by alias (`date`); part of the generation-cache key, so editing a label invalidates cached generations only for that item |
| `GroundTruthSet.sha256` | `sha256("\n".join(sorted(canonical_json(item) for item in items)))`: independent of file order |

`canonical_json` = JSON with sorted keys, separators `,` and `:`, no ASCII escaping.

### 2.4 Set operations (`GroundTruthSet`)

| Method | Behaviour | Status |
|---|---|---|
| `by_id()`, `facts()` | index by id; group by fact_id | now |
| `filter(splits=("test",), require_reviewed=False, exclude_ids=())` | `"all"` keeps both splits | now |
| `coverage_table()` | item counts per (lang_script, domain, answerable) | now |
| `subsample(fraction, seed, min_per_stratum=20)` | keep a fact iff `u(seed, fact_id) < fraction`, where `u` = first 8 bytes of `sha256(f"{seed}:{fact_id}")` scaled to [0, 1); then add facts in ascending `u` to every (lang_script, domain, answerable) stratum below `min_per_stratum`. Deterministic, independent of file order, stable when items are appended. The same ids are used at every checkpoint (`subsample_ids.json`) | M3 |
| `dataset_stats(gt)` | coverage, per-stratum CI half-widths, design effect, MDE (`slm-forge data stats`) | M5 |

## 3. Scoring semantics

| `answerable` | `allow_abstention` | Model abstains | Model rejects the premise | Wrong / asserted answer | Right answer |
|---|---|---|---|---|---|
| true | true (default) | ABSTAIN_ACCEPTED (0) | false denial → MISMATCH → wrong() | CONFABULATION / WRONG_LOW_CONF / WRONG_HEDGED | CORRECT / CORRECT_HEDGED |
| true | false | ABSTAIN_UNWARRANTED (−0.5) | same as above | same as above | same as above |
| false | must be true (E05) | ABSTAIN_CORRECT (+1) | ABSTAIN_CORRECT (+1) | any asserted answer → wrong() | n/a |

- `wrong()` = WRONG_HEDGED if hedged, else CONFABULATION if confidence ≥ τ or unknown, else WRONG_LOW_CONF.
- Abstaining never counts as correct for `factual_accuracy`; abstaining on an answerable item is never an error either.
- Use `allow_abstention: false` only for facts a deployed model is *required* to know (core product terms, statutory rates). Full decision table: `docs/metrics.md` §4.

## 4. Validation rules

### 4.1 Errors

| Code | Rule | Status |
|---|---|---|
| E01 | Invalid JSON; wrong type; unknown top-level or nested key; a field constraint not covered by another code (e.g. a negative tolerance); `domain` not in `allowed_domains` (when loaded with a config) | now |
| E02 | `id` does not match the pattern; duplicate `id` | now |
| E03 | `expected_answer` key missing, or empty/whitespace (use `null` for unanswerable) | now |
| E04 | Explicit `answerable` disagrees with `expected_answer` | now |
| E05 | Unanswerable item without `unanswerable_reason`; with `aliases`, `numeric`, `date` or `citation`; or with `allow_abstention: false` | now |
| E06 | Answerable item with `unanswerable_reason` or `premise_rejection_cues` | now |
| E07 | fr/en item with `script: arab`; Darija item without `script` | now |
| | Declared script contradicted by detection on the prompt at ≥ 0.9; Darija script detected when absent | M1 |
| E08 | numeric/percent/currency/duration without `numeric`; percent with a unit other than `%` | now |
| | currency unit ∉ {MAD, EUR, USD}; duration unit ∉ {days, months, years}; the value parsed from `expected_answer` is not within tolerance of `numeric.value` | M1 |
| E09 | date without `date`; citation without `citation`; choice whose `choices` do not contain `expected_answer` | now |
| | boolean `expected_answer` that does not map to exactly one polarity (stored as canonical yes/no) | M1 |
| E10 | Normalized `known_false` overlaps normalized {expected_answer} ∪ aliases, or a numeric distractor falls within tolerance of the gold | M1 |
| E11 | Duplicate `(lang_script, L1(prompt))` | M1 |
| E12 | Members of a fact_id disagree on `answerable`, `domain` or `answer_type` | now |
| | … or on the canonical typed value | M1 |
| E13 | A fact_id appears in both `dev` and `test` | now |
| E14 | Empty alias | now |
| | `expected_answer` or alias empty after L1 normalization | M1 |

### 4.2 Warnings

| Code | Rule | Status |
|---|---|---|
| W01 | Legacy record upgraded; derived id | now |
| W02 | Answerable Darija item with no alias in the other script | M1 |
| W03 | `free_text` answer that parses as a number or date (use a typed `answer_type`) | M1 |
| W04 | `expected_answer` longer than 12 tokens | M1 |
| W05 | The gold value appears in the prompt (prompt-echo risk) | M1 |
| W06 | Answerable finance/legal item without `source.as_of`, or with `as_of` older than 18 months | M1 |
| W07 | A (lang_script, domain, answerable) cell with n < 30 | M1 |
| W08 | Unanswerable share per lang_script outside [0.15, 0.50] | M1 |
| W09 | Fewer than 30% of numeric items have `known_false` | M1 |
| W10 | A fact_id with a single language member | M1 |
| W11 | Weak disagreement between declared and detected script | M1 |
| W12 | `reviewed: false` items present | M1 |
| W13 | Rule langid disagrees with the declared language (an English prompt labelled `fr` triggers it) | M1 |

Issues are reported as `LEVEL CODE [line N]: message`, collected per line instead of stopping at
the first error.

## 5. Examples

One line each (they validate with the current loader). All facts are fictional: *Banque Atlas
Fictive*, *Compte Épargne Zénith*, *Prêt Atlas Habitat* and *Law No. 77-19* do not exist.

English, currency:

```json
{"schema_version":1,"id":"doc-fin-en-0001","fact_id":"doc-zenith-deposit","prompt":"What is the minimum opening deposit for the Compte Épargne Zénith at Banque Atlas Fictive?","language":"en","domain":"finance","expected_answer":"500 MAD","answer_type":"currency","numeric":{"value":500,"unit":"MAD"},"aliases":["500 dirhams","500 DH"],"known_false":["1000 MAD","250 MAD"],"anchor_terms":["Compte Épargne Zénith"],"source":{"citation":"fictional tariff sheet","as_of":"2026-01-01"}}
```

French, date:

```json
{"schema_version":1,"id":"doc-fin-fr-0001","fact_id":"doc-zenith-launch","prompt":"À quelle date la Banque Atlas Fictive a-t-elle lancé le Compte Épargne Zénith ?","language":"fr","domain":"finance","expected_answer":"12 mai 2019","answer_type":"date","date":{"value":"2019-05-12"},"aliases":["12/05/2019","2019-05-12"],"known_false":["12 mai 2018"],"source":{"citation":"fictional press release","as_of":"2026-01-01"}}
```

Darija, Arabic script, citation:

```json
{"schema_version":1,"id":"doc-leg-dar-a-0001","fact_id":"doc-coop-article","prompt":"شمن فصل ف القانون رقم 77-19 ديال التعاونيات الرقمية أطلس كيحدد العدد الأدنى ديال الأعضاء المؤسسين؟","language":"darija","script":"arab","domain":"legal","expected_answer":"الفصل 14","answer_type":"citation","citation":{"article":"14","instrument":"77-19","instrument_aliases":["القانون 77-19","loi 77-19","l9anoun 77-19"]},"aliases":["المادة 14","l-fasl 14","article 14"],"known_false":["الفصل 41"],"anchor_terms":["77-19"],"source":{"citation":"fictional Law 77-19","as_of":"2026-01-01"}}
```

Darija, Latin script (Arabizi), code-switched, percent:

```json
{"schema_version":1,"id":"doc-fin-dar-l-0001","fact_id":"doc-atlas-ltv","prompt":"ch7al howa l'LTV l9oswa dyal Prêt Atlas Habitat?","language":"darija","script":"latn","code_switch":["fr"],"domain":"finance","expected_answer":"70%","answer_type":"percent","numeric":{"value":70,"unit":"%"},"aliases":["70 f lmya","sb3in f lmya","سبعين فالمية"],"known_false":["80%","60%"],"anchor_terms":["Prêt Atlas Habitat"],"source":{"citation":"fictional product sheet","as_of":"2026-01-01"}}
```

Unanswerable (false premise):

```json
{"schema_version":1,"id":"doc-leg-en-u-0001","fact_id":"doc-coop-abolish-u","prompt":"Why did Law 77-19 abolish Atlas digital cooperatives in 2021?","language":"en","domain":"legal","expected_answer":null,"answer_type":"free_text","unanswerable_reason":"false_premise","premise_rejection_cues":["did not abolish","does not abolish"]}
```

In a real file, translations of one fact share one `fact_id` (for example `doc-atlas-ltv` in en, fr,
darija_arab and darija_latn); W10 warns about single-language facts.

## 6. Legacy upgrade

A README-format record `{prompt, expected_answer, language, domain?, allow_abstention?}` (no
`schema_version`, no `id`) is accepted (now):

| Field | Upgraded value |
|---|---|
| `id` | `"legacy-" + sha256(casefolded, whitespace-collapsed prompt + "\|" + language)[:12]` |
| `fact_id` | = `id` |
| `tags` | `+ ["legacy"]` |
| everything else | the field defaults (`split: test`, `domain: general`, `allow_abstention: true`, `reviewed: true`) |
| issue | W01 |

`expected_answer` stays a required key (E03), and Darija legacy records still need `script` (E07)
until M1 detects it. `answer_type` inference ("70%" → percent) and `slm-forge data validate
--upgrade OUT` (writes the upgraded v1 file) arrive in M1.

## 7. Companion files

| File | Content | Status |
|---|---|---|
| `data/eval/ground_truth.sample.jsonl` | 24 fictional items, 6 per core bucket, 25% unanswerable; used by tests, docs and `configs/tiny-cpu-test.yaml` | now |
| `data/eval/labels/responses_gold.jsonl` | Human labels of *real* model responses for `validate-detectors`: `{item_id, response, response_sha256, source: {model, checkpoint_step \| null}, labels: {verdict: match \| mismatch \| abstain \| premise_rejection \| asserted \| non_answer \| unclear, hedged: bool, language_ok: bool}, annotators: [str], adjudicated: bool, split}`; ≥ 600 responses stratified by lang_script × outcome, 20% double-labelled (`docs/annotation.md`) | Track D3 / M7 |
| `tests/fixtures/scoring_cases.jsonl` | `{case_id, item, response, token_logprobs \| null, expected: {outcome, verdict, hedged, abstain_cue, answered_signal}, note}`; ≥ 150 en/fr cases at M1, ≥ 400 (≥ 80 per core bucket) at M4, including adversarial ones ("I don't know why people say 60%; it's 70%", "70 %, pas 80 %", "ma kayn ta mochkil, l'taux huwa 4%", "70%، والله أعلم") | M1 / M4 |
| `runs/_shared/verdicts.sqlite` | Human and judge verdicts keyed by (item id, L1 response) | M7 |

Teacher-distilled output never enters any of these files (`data.distill.assert_not_distilled`).
