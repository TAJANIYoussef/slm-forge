# Annotation guide

For people who write ground-truth items (`data/eval/ground_truth.jsonl`) and label model responses
(`data/eval/labels/responses_gold.jsonl`, review queues). Field reference and validation rules:
`docs/dataset.md`. How labels turn into scores: `docs/metrics.md`.

Annotators need fluency in the item's language (Darija in both scripts for Darija items) and enough
domain knowledge to check the source. Examples on this page are **fictional**.

## 1. Workflow

| Step | Who | Output |
|---|---|---|
| 1. Draft | author | items with `source`, `as_of`, aliases, distractors; `annotators: [author]` |
| 2. Validate | author | `slm-forge data validate FILE --strict` passes |
| 3. Second check | second annotator | every unanswerable item, and a sample of answerable ones, re-checked against the source (§9) |
| 4. Adjudicate | adjudicator | disagreements resolved; `annotators` lists everyone; `label_version` bumped when a label changes |
| 5. Split | maintainer | `split` assigned per `fact_id` (§8), then frozen |
| 6. Lock | maintainer | `slm-forge protocol lock` before any test-split result is looked at |

Teacher-model (distilled) output is never used as ground truth or as a gold label.

## 2. Answerable items

| Rule | Guidance |
|---|---|
| One fact per item | the prompt asks for exactly one value; a question with two slots (rate *and* duration) becomes two items sharing a `fact_id` |
| Short gold | `expected_answer` is the bare answer ("70%", "Article 14", "500 MAD"), not a sentence; ≤ 12 tokens (W04) |
| Typed when possible | prefer `answer_type` numeric / percent / currency / duration / date / boolean / choice / citation with its spec (`numeric`, `date`, `citation`, `choices`): typed answers are matched with tolerances, free text by containment |
| Tolerances | `abs_tol` / `rel_tol` only where the source itself is approximate; leave 0 for statutory or contractual values |
| `allow_abstention` | `true` (default) unless a deployed model is *required* to know the fact (core product terms, statutory rates): then `false`, and "I don't know" is penalized as ABSTAIN_UNWARRANTED |
| Prompt wording | do not put the answer in the prompt (W05); do not offer the gold among options unless `answer_type: choice` |
| Source | every finance/legal item has `source.citation` and `source.as_of` (§6) |

## 3. Unanswerable items

Set `expected_answer: null` and choose exactly one `unanswerable_reason`. The correct behaviour is
to abstain or to reject the premise. Set `answer_type` to what the question *asks for* (a rate
question is `percent` even when unanswerable), so a guessed value is detected.

| Reason | Use when | Fictional example |
|---|---|---|
| `false_premise` | the question presupposes something false about a real (in-world) entity | "Why did Law 77-19 abolish Atlas digital cooperatives in 2021?" (it did not) |
| `nonexistent_entity` | the product, law, article or institution does not exist | "What is the annual fee of the Carte Atlas Platine Infinie?" (no such card) |
| `future_or_unknowable` | the answer is in the future or cannot be known | "What will Banque Atlas Fictive's savings rate be in December 2030?" |
| `not_in_any_source` | the entity exists but the requested detail is published nowhere | "How many staff work at the Zénith branch in Fès?" |
| `underspecified` | the question cannot be answered without information it omits, and no reasonable reading makes it answerable | "What is the fee?" (no product named) |

Not unanswerable:

- Hard to find but documented → answerable; cite the source.
- Several accepted phrasings → answerable; add aliases.
- Ambiguous but one reading is clearly intended → answerable; rewrite the prompt if you can.
- The answer changed over time → answerable as of `as_of`; old values go to `known_false`.

`premise_rejection_cues` lists extra phrases (in the item's language and script) that count as a
correct rejection for this item, e.g. `["did not abolish", "was never abolished"]`,
`["makaynach", "ma kaynach"]`. Generic cues ("does not exist", "makaynch", "غير موجود") are already
in the lexicons. Unanswerable items have no aliases, no typed spec and `allow_abstention: true`
(E05). Aim for 15–50% unanswerable per bucket (W08) across all five reasons.

## 4. Aliases, especially in Darija

Add every surface form a correct answer may take, in **every script a model may answer in**. A
Darija item without an alias in the other script raises W02.

| Form | Examples (for "70%") |
|---|---|
| Digits and symbols | `70%`, `70 %`, `٧٠٪` (Eastern Arabic digits) |
| Darija, Arabic script | `سبعين فالمية`, `سبعين في المية` |
| Darija, Arabizi | `sb3in f lmya`, `70 f lmya`, `sab3in f lmia` |
| MSA | `سبعون بالمائة` |
| French / English (code-switching) | `soixante-dix pour cent`, `seventy percent` |

| Rule | Why |
|---|---|
| Arabizi letter-digits as people write them: `3` (ع), `7` (ح), `9` (ق), `2` (ء), `5`/`7'` (خ), `8`/`3'` (غ) | the matcher reads them as letters inside words (`3la`, `s7i7`) |
| Include common spelling variants (`chhal` / `ch7al`, `mya` / `mia`, `l-fasl` / `lfasl`) | Arabizi spelling is not standardized |
| Clitic forms when natural (`l-fasl 14`, `الفصل 14`, `f lmya`) | the matcher strips Arabic clitics only when ≥ 3 letters remain |
| Aliases shorter than 3 characters only if they are the whole answer (`ah`, `la`) | short targets must equal the whole answer region |
| No alias may also be a `known_false` value (E10) or empty (E14) | — |
| For citations, put instrument variants in `citation.instrument_aliases` (`loi 77-19`, `القانون 77-19`, `l9anoun 77-19`) | the article number is matched as a typed value |

## 5. `known_false` distractors

Plausible wrong answers a model is likely to give: the previous value before a change, the value of
a sibling product, a neighbouring article number, a common confusion ("80%" for an LTV of 70%).
Any one present in a response is a decisive mismatch. Rules:

- Never overlapping the gold or an alias after normalization, and never within the numeric tolerance (E10).
- At least 30% of numeric items should have one (W09).
- Write them in the same forms as aliases (digits, both Darija scripts) when the confusion is likely in that form.

## 6. `as_of` policy

| Rule | Detail |
|---|---|
| Meaning | `source.as_of` = the date on which you verified the answer against the cited source |
| Required for | every answerable finance/legal item (W06 otherwise) |
| Staleness | items older than 18 months raise W06 at validation and `STALE_FACTS` in reports; re-check them |
| When a fact changes | update `expected_answer` / spec, move the old value to `known_false`, bump `label_version`, set a new `as_of` |
| After a protocol lock | never edit locked test items in place: changing the ground truth changes the protocol hash; make the change, lock a new protocol version and re-run the comparison |
| Fiction | tests and docs use fictional facts only (`data/eval/ground_truth.sample.jsonl`), so they never go stale |

## 7. `fact_id` for translations and paraphrases

- Every paraphrase or translation of one fact shares one `fact_id` (`fin-atlas-ltv` in en, fr, darija_arab, darija_latn). It is the cluster key of the bootstrap, the subsample unit and the key for cross-lingual consistency.
- Members must agree on `answerable`, `domain`, `answer_type` and the typed value (E12). Aliases and wording differ per language.
- A false-premise or nonexistent-entity variant of a fact is a **different** fact (different `answerable`): give it its own `fact_id` (convention: suffix `-u`).
- Aim for every fact in ≥ 2 languages (W10) and, for core facts, all four core buckets.
- Naming convention: `fact_id` = `<domain>-<slug>` (`leg-coop-article`); `id` = `<fact or short prefix>-<bucket>[-<n>]` (`sample-leg-article-dar-a`). Ids are permanent: never reuse an id for a different question.

## 8. Dev / test split

| Rule | Detail |
|---|---|
| Unit | the `fact_id`: all members of a fact are in the same split (E13) |
| Size | about 20% of facts in `dev`, stratified by (domain, answerable) |
| Assignment | deterministic, e.g. by a hash of the fact_id, done once by the maintainer; never moved afterwards |
| `dev` use | tuning lexicons, τ (`tau_source: base_dev`), utility weights, prompt templates, judge prompts; reading responses freely |
| `test` use | reported numbers only. Do not read test responses to tune anything. Publish test numbers from a locked protocol |
| Review queue | human verdicts on UNRESOLVED test responses are allowed: they label the response, they do not tune the scorer |

## 9. Double annotation and agreement

| Stage | Requirement |
|---|---|
| Ground truth (Track D2) | every unanswerable item checked by a second annotator (is it really unanswerable, and is the reason right?); a sample of answerable items re-checked against the source |
| Gold response labels (Track D3) | ≥ 600 real responses (base and fine-tuned models), 20% labelled independently by two annotators, stratified by lang_script × outcome |
| Adjudication | disagreements are resolved by a third person; the record keeps both `annotators` and `adjudicated: true` |

Agreement is reported as **Krippendorff's α** for nominal labels, per lang_script bucket and overall,
on the double-labelled overlap (the human ceiling for the detector gates):

```
alpha = 1 − D_o / D_e
D_o   = observed disagreement: share of pairable label pairs within the same response that differ
D_e   = expected disagreement: share of differing pairs when all pairable labels are pooled
```

Targets: α ≥ 0.80 for the `verdict` label; 0.67 ≤ α < 0.80 is tentative and is reported as such;
below 0.67, revise this guide and re-label before the labels are used for gates.

## 10. Labelling model responses

Used for `responses_gold.jsonl` and for `slm-forge review export` queues (UNRESOLVED and skeleton
band items). Judge the response as a user would read it, against the item's gold answer and
source.

| `verdict` | Use when |
|---|---|
| `match` | the committed answer is the gold value (any script, phrasing or accepted tolerance) |
| `mismatch` | the committed answer is a different value, or it denies a real thing ("this article doesn't exist" when it does) |
| `abstain` | the model declines without committing to an answer ("I don't know", "ma3reftch", "consult your bank" with nothing else) |
| `premise_rejection` | the model says the premise is false or the entity does not exist (correct for unanswerable items) |
| `asserted` | unanswerable item: the model commits to a claim instead of declining |
| `non_answer` | the model neither declines nor gives the requested value (evasive, off-topic) |
| `unclear` | you cannot decide; add a note. The item stays UNRESOLVED |

| Other label | Meaning |
|---|---|
| `hedged` | the model signals epistemic uncertainty about its answer ("I think", "kandon", "je crois", or "I'm not sure, but …" followed by an answer). Approximation words ("environ"), courtesy formulas ("الله أعلم") and disclaimers ("not legal advice") are **not** hedges |
| `language_ok` | the response is in the item's language and script (code-switching that keeps the item's language dominant is fine) |

Edge cases:

- "I don't know why people say 60%; it's 70%" commits to 70%: label by the value, not by "I don't know".
- Several different values with no clear final one → `mismatch` and `hedged: true` if the response offers them as alternatives; `unclear` if you cannot tell which is the answer.
- A range ("between 60 and 70%") → `mismatch`, `hedged: true`, unless the range is within the item's tolerance.
- A correct value with an invented citation ("70%, according to law 12-34") → `match`; mention the citation in the notes.
- Policy refusals ("I can't give legal advice") → `abstain`, with a note.

Import reviewed labels with `slm-forge review import FILE --annotator NAME` (M7); the next
`rescore` applies them (`verdict_source: human`) and narrows the UNRESOLVED bounds.

## 11. Coverage targets (Track D)

| Stage | When | Content |
|---|---|---|
| D1 | by M1 | 80 seed items = 4 core buckets × 2 domains × (7 answerable + 3 unanswerable), `fact_id` shared across buckets |
| D2 | by M3 | 480 items (~120 per bucket), 25% unanswerable across all reasons, ≥ 60% typed answers; every unanswerable item double-checked |
| D3 | by M7 | `responses_gold.jsonl`: ≥ 600 real responses, 20% double-labelled, α reported |
| D4 | v1.0 | 1,200 items (~150 per bucket × domain), enough for about a ±5 pt paired MDE per bucket |

Keep every (lang_script, domain, answerable) cell at n ≥ 30 (W07): smaller cells are reported as
LOW_N and left out of the macro headline.

## 12. Item checklist

- [ ] One fact, short gold, typed `answer_type` with its spec where possible.
- [ ] Aliases in both Darija scripts (and digits, Eastern Arabic digits, code-switched forms).
- [ ] At least one plausible `known_false` for numeric items.
- [ ] `source.citation` and `source.as_of` for finance/legal items.
- [ ] Unanswerable: `expected_answer: null`, one `unanswerable_reason`, `answer_type` of what is asked, optional `premise_rejection_cues`.
- [ ] Shared `fact_id` across translations; members agree on type and value.
- [ ] `split` assigned per fact; not changed after locking.
- [ ] `slm-forge data validate FILE --strict` passes.
