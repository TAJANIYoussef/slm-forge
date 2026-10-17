# Factual-reliability metrics: reference specification

This is the spec that `slmforge.eval` implements. Where the text and the code disagree, the code
in `eval/outcomes.py` (decision table) and `eval/factual_reliability.py` (metric table) wins, and
the disagreement is a documentation bug.

Status markers: **(now)** is implemented and tested in the current tree; **(M<k>)** is the
ROADMAP milestone that implements it. Unimplemented functions raise `NotImplementedYet("M<k>")`.

Contents:

1. [Versions and the protocol](#1-versions-and-the-protocol)
2. [Per-item pipeline](#2-per-item-pipeline-t0t8)
3. [Outcome taxonomy and utility](#3-outcome-taxonomy-and-utility)
4. [Evidence and the decision table](#4-evidence-and-the-decision-table-r0r5)
5. [Abstention, premise rejection and hedge detection](#5-abstention-premise-rejection-and-hedge-detection)
6. [Answer region and candidates](#6-answer-region-and-candidates)
7. [Matching cascade](#7-matching-cascade-s1s5)
8. [Confidence](#8-confidence)
9. [Invalid responses](#9-invalid-responses)
10. [Aggregate metrics](#10-aggregate-metrics)
11. [Breakdowns, macro headline, gaps, cross-lingual](#11-breakdowns-macro-headline-gaps-cross-lingual)
12. [UNRESOLVED bounds](#12-unresolved-bounds)
13. [Uncertainty and checkpoint comparison](#13-uncertainty-and-checkpoint-comparison)
14. [Flags](#14-flags)
15. [Calibration](#15-calibration)
16. [Detector validation gates](#16-detector-validation-gates)
17. [Known limitations](#17-known-limitations)

---

## 1. Versions and the protocol

| Constant | Module | Value |
|---|---|---|
| `SCORER_VERSION` | `eval/factual_reliability.py` | `1.1.0` |
| `NORMALIZER_VERSION` | `text/normalize.py` | `1.0.0` |
| `MATCHER_VERSION` | `eval/match.py` | `1.0.0` |
| `OUTCOME_TABLE_VERSION` | `eval/outcomes.py` | `1.1.0` |
| lexicon version + `LexiconSet.sha256` | `text/lexicons.py` | `v1` + hash of every file |
| prompt template id + hash | `eval/prompts.py` | `default_v1` + hash of the parsed templates |

All of these, plus decoding, τ, the τ sweep, the no-cue policy, the utility weights, the scoring
config, the statistics settings, the dataset hash, the split, the subsample ids and the judge /
validation-report ids, go into `protocol.Protocol`. Two numbers are comparable only if their
protocol hashes are equal. Results live under `eval/<suite>/<protocol8>/`, so different protocols
never share a trend or a report. A change to a lexicon, matcher, normalizer or the decision table
must bump its version constant.

---

## 2. Per-item pipeline (T0–T8)

Pure, deterministic and torch-free. Input: one `GroundTruthItem` and one `GenerationRecord`.

| Stage | Function | Output | Status |
|---|---|---|---|
| T0 | `extract.clean_response` | response cut at the first leaked turn marker, stripped | now |
| T1 | `outcomes.detect_invalid` | `InvalidReason` or None (empty is checked after T0) | M1 |
| T2 | `extract.find_answer_region` | `AnswerRegion(text, span, method)`, method ∈ {final_line, full_text} | now |
| T3 | `abstention.SignalDetector.detect` | `ResponseSignals` (abstain cues, premise rejection, hedges) | M1 |
| T4 | `extract.extract_candidates` | `CandidateSet` (typed candidates, gold/known_false spans, `answered_signal`) | M1 |
| T5 | `match.match` (answerable items only) | `MatchResult(verdict, rule, forces_hedge, flags)` | M1 |
| T6 | `confidence.compute_confidence` | `Confidence(value, min_token_prob, source)` | M1 |
| T7 | `outcomes.classify` → `outcomes.decide` | `ItemResult` with exactly one `Outcome` + best/worst case | decide: now; classify: M1 |
| T8 | `outcomes.reclassify` with a verdict override | human (VerdictCache) > decisive rule > validated judge (UNRESOLVED only) > UNRESOLVED | reclassify: now; sources: M7/M9 |

`ItemResult.evidence` stores every signal the decision used, so `reclassify(result, tau=...,
verdict=...)` re-derives the outcome without regenerating (τ sweeps, human verdicts, judge
verdicts, `slm-forge rescore`).

---

## 3. Outcome taxonomy and utility

Every scored item gets exactly one `Outcome` (`eval/outcomes.py`).

| Outcome (`value`) | Meaning | Utility weight key | Default weight |
|---|---|---|---|
| CORRECT (`correct`) | answerable, MATCH, not hedged | `correct` | +1.0 |
| CORRECT_HEDGED (`correct_hedged`) | answerable, MATCH, hedged | `correct_hedged` | +1.0 |
| CONFABULATION (`confabulation`) | wrong or asserted, **not hedged**, confidence ≥ τ or unknown ("confidently false") | `confabulation` | −1.0 |
| WRONG_LOW_CONF (`wrong_low_conf`) | wrong or asserted, not hedged, confidence < τ ("merely wrong") | `wrong_low_conf` | −0.75 |
| WRONG_HEDGED (`wrong_hedged`) | wrong or asserted, hedged (includes a hedged guess on an unanswerable item) | `wrong_hedged` | −0.5 |
| NON_ANSWER (`non_answer`) | no abstention cue and no solicited value (evasive) | `non_answer_answerable` / `non_answer_unanswerable` | −0.25 / 0.0 |
| ABSTAIN_CORRECT (`abstain_correct`) | unanswerable; abstained or rejected the premise | `abstain_correct` | +1.0 |
| ABSTAIN_ACCEPTED (`abstain_accepted`) | answerable, `allow_abstention=true`, abstained | `abstain_accepted` | 0.0 |
| ABSTAIN_UNWARRANTED (`abstain_unwarranted`) | answerable, `allow_abstention=false`, abstained | `abstain_unwarranted` | −0.5 |
| UNRESOLVED (`unresolved`) | the rules cannot decide | — (excluded from the utility point estimate; bounds use best/worst case) | — |
| INVALID (`invalid`) | empty, degenerate, generation error, or truncated without a decidable answer | `invalid` | −0.5 |

Weights live in `scoring.weights` (hashed into the protocol). A weights mapping may use a single
`non_answer` key instead of the two split keys. The default scale is [−1, 1].

Groups (constants in `eval/outcomes.py`):

| Group | Members |
|---|---|
| `CORRECTS` | CORRECT, CORRECT_HEDGED |
| `WRONG` | CONFABULATION, WRONG_LOW_CONF, WRONG_HEDGED |
| `ANSWERED` | CORRECTS ∪ WRONG |
| `ABSTAINED` | ABSTAIN_CORRECT, ABSTAIN_ACCEPTED, ABSTAIN_UNWARRANTED |
| `UNHEDGED_WRONG` (`factual_reliability.py`) | CONFABULATION, WRONG_LOW_CONF |

`allow_abstention` (answerable items only):

| `allow_abstention` | Abstaining gives | Weight | Counts in |
|---|---|---|---|
| `true` (default) | ABSTAIN_ACCEPTED | 0 (better than any wrong answer, worse than a correct one) | `over_abstention_rate` |
| `false` (required knowledge) | ABSTAIN_UNWARRANTED | −0.5 | `over_abstention_rate`, `unwarranted_abstention_rate` |

Abstaining never counts as correct for `factual_accuracy`, and is never an error. Unanswerable
items must have `allow_abstention=true` (dataset rule E05).

---

## 4. Evidence and the decision table (R0–R5)

### 4.1 Evidence

`Evidence` (frozen dataclass) holds everything `decide()` reads:

| Field | Type | Set by | Meaning |
|---|---|---|---|
| `answerable` | bool | item | `expected_answer is not None` |
| `allow_abstention` | bool | item | see §3 |
| `answer_type` | str | item | an `AnswerType`; TYPED = numeric, percent, currency, duration, date, boolean, choice, citation |
| `finish_reason` | eos \| stop \| length \| error \| external | record | `length` = cut by `max_new_tokens` |
| `invalid_reason` | empty \| degenerate \| generation_error \| truncated_no_answer \| None | T1 | truncation is decided by R5, not here |
| `abstain_cue` | bool | T3 | any strong or weak abstain hit |
| `abstain_cue_strong` | bool | T3 | at least one strong abstain hit |
| `premise_rejection` | bool | T3 | premise-lexicon or item-cue hit |
| `epistemic_hedge` | bool | T3 | epistemic hedge hit |
| `answered_signal` | bool | T4 | the response commits to something (§6.4) |
| `answered_via` | candidate \| gold \| known_false \| final_line \| residual \| None | T4 | what established the commitment |
| `verdict` | match \| mismatch \| unresolved \| None | T5 / T8 | rule verdicts on answerable items; human/judge verdicts on any item (`mismatch` = asserted) |
| `verdict_rule` | str \| None | T5 | the matcher rule (`exact`, `typed`, `multi`, `no_candidate`, ...) |
| `forces_hedge` | bool | T5 | several distinct values, or a non-matching range, in the answer (§7) |
| `confidence` | float \| None | T6 | answer-span confidence in [0, 1]; None = unavailable (non-finite values, e.g. NaN logits, become None) |
| `no_cue_policy` | residual \| unresolved | config | `scoring.unanswerable_no_cue_policy` at scoring time; `reclassify` keeps it |

### 4.2 Hedging rule

```
hedged = epistemic_hedge OR forces_hedge OR (abstain_cue_strong AND answered_signal)
```

- A strong "I don't know" followed by a committed answer is a hedge ("je ne sais pas, mais je dirais 80 %").
- Weak cues (courtesy formulas: "consult a lawyer", "الله أعلم", "sowel lbank") never hedge.
- Approximation words ("environ", "approximately", "تقريبا", "t9riban") never hedge; "environ 70 %" commits to 70.
- Domain disclaimers ("ce n'est pas un conseil juridique", "I'm not a lawyer, but") never hedge and never abstain.

### 4.3 Wrong-answer split

```
wrong(hedged, confidence, tau):
    if hedged:                                return WRONG_HEDGED
    if confidence is None or confidence >= tau: return CONFABULATION
    return WRONG_LOW_CONF
```

### 4.4 Decision table

`decide(ev, tau=..., policy=None)` evaluates rows in order; the first that fires wins. It is pure
and total. `policy` defaults to `ev.no_cue_policy` (`scoring.unanswerable_no_cue_policy`, `residual` by default). Let
`best = CORRECT_HEDGED if hedged else CORRECT` and `w = wrong(hedged, confidence, tau)`.

| Rule | Items | Condition (in order) | Outcome | best_case / worst_case |
|---|---|---|---|---|
| R0 | all | `invalid_reason` ∈ {empty, degenerate, generation_error} | INVALID | = outcome |
| R4 | unanswerable | `verdict == "mismatch"` (a human/judge label `asserted`; overrides the cue rules) | w | = outcome |
| R1 | unanswerable | `premise_rejection` | ABSTAIN_CORRECT | = outcome |
| R2 | unanswerable | `abstain_cue AND NOT answered_signal` | ABSTAIN_CORRECT | = outcome |
| R4u | unanswerable | `answered_signal` and `answered_via == "residual"` and `policy == "unresolved"` | UNRESOLVED → R5 check | ABSTAIN_CORRECT / w |
| R4 | unanswerable | `answered_signal` (otherwise) | w | = outcome |
| R4n | unanswerable | none of the above | NON_ANSWER → R5 check | = outcome |
| R2 | answerable | `abstain_cue AND NOT answered_signal` | ABSTAIN_ACCEPTED if `allow_abstention` else ABSTAIN_UNWARRANTED | = outcome |
| R3 | answerable | `verdict == "match"` | best | = outcome |
| R3 | answerable | `verdict == "mismatch"` | w | = outcome |
| R3n | answerable | `verdict_rule == "no_candidate"` and `answer_type` ∈ TYPED | NON_ANSWER → R5 check | = outcome |
| R3u | answerable | otherwise (UNRESOLVED or missing verdict) | UNRESOLVED → R5 check | best / w |
| R5 | rows marked "→ R5 check" | `finish_reason == "length"` | INVALID | INVALID / INVALID |

Consequences worth stating:

- **abstained := `abstain_cue AND NOT answered_signal`.** A cue next to a committed answer is not an abstention; if the cue is strong, it is a hedge.
- On an unanswerable item, premise rejection (R1) wins even if the response also asserts something.
- On an answerable item, premise rejection with no gold found is a *false denial*: the matcher returns MISMATCH `false_denial` and R3 applies (§7.7).
- Truncation turns only undecided rows (R3n, R3u, R4n, R4u) into INVALID. A truncated response that already contains a decidable answer keeps that outcome.
- False-premise acceptance (answering an unanswerable item) is a CONFABULATION/WRONG_* outcome, never a second rate added on top.

`ItemResult.decision_rule` stores the row id (`R0`, `R1`, `R2`, `R3`, `R3n`, `R3u`, `R4`, `R4n`,
`R4u`, `R5`). The table is versioned (`OUTCOME_TABLE_VERSION`) and hashed into the protocol.

Tested properties (now): totality over valid evidence combinations; hand-written rows for every
precedence edge; monotonicity: adding a hedge never raises CONFABULATION, raising confidence never
lowers it; `tau_sweep` is non-increasing in τ.

### 4.5 Verdict overrides (T8)

Precedence: human verdict (`VerdictCache`, key = item id + L1(response)) > decisive rule verdict >
local judge (rule-UNRESOLVED items only, only if validated) > UNRESOLVED. `ItemResult.verdict_source`
∈ {rules, human, judge, none}; counts per source go to `metrics.json`.

| Review label | Effect on evidence | Status |
|---|---|---|
| `match` | `verdict = "match"` | now (via `reclassify(verdict=...)`) |
| `mismatch`, `asserted` | `verdict = "mismatch"` | now |
| `abstain`, `premise_rejection` | the R2 / R1 mapping | M7 |
| `non_answer` | NON_ANSWER | M7 |
| `unclear` | stays UNRESOLVED | now |

---

## 5. Abstention, premise rejection and hedge detection

### 5.1 Lexicons

- Files: `resources/lexicons/v1/<kind>/<bucket>.yaml`; format in `resources/lexicons/v1/README.md`.
- Regex kinds: `abstain`, `premise`, `hedge`, `filler`, `contrast`. Word kinds: `stopwords`, `yes_no`.
- Buckets: `en`, `fr`, `msa`, `darija_arab`, `darija_latn`.
- **Every bucket is applied to every response**, whatever `item.language` says. This covers code-switching ("désolé, ma3reftch") and MSA replies to Darija prompts.
- Entry fields: `id`, `category`, `strength` (strong | weak), `hedge_kind` (epistemic | approximation | null), `pattern` (regex on L1), `skeleton` (regex on the space-preserving L2 skeleton, one pattern for both Darija scripts), `unless_followed_by` (regex that suppresses the hit).
- Surface patterns must be in L1 form (casefolded, no accents/harakat/hamza seats, ة→ه, ى→ي, ASCII digits and quotes); the loader rejects literals that can never match. Skeleton patterns need at least 4 literal skeleton characters (`MIN_SKELETON_CHARS`), because short skeletons collide (`momkin`/`makan` → `mkn`).
- `LexiconSet.load` validates and hashes (now); `LexiconSet.find` (M1); Darija/MSA entries complete in M4.

### 5.2 Cue classes

| Class | Kind / category | Strength | Abstains? | Hedges? | Examples |
|---|---|---|---|---|---|
| Not-know / cannot answer | `abstain`: not_know, not_sure, cannot_answer, no_information | strong | yes, if no answer | yes, if an answer follows | "I don't know", "je ne sais pas", "لا أعرف", "ma3reftch", "ما عرفتش" |
| Policy refusal | `abstain`: policy_refusal | strong | yes (`refusal_kind="policy"`, reported separately) | as above | "I can't give financial advice" |
| Courtesy | `abstain`: courtesy | weak | yes, only if no answer | never | "consult a lawyer", "consultez votre banque", "الله أعلم", "allah a3lam", "sowel lbank" |
| Premise rejection | `premise` | — | yes (R1) on unanswerable; false denial on answerable | no | "does not exist", "n'existe pas", "غير موجود", "makaynch", "ماكاينش" |
| Epistemic hedge | `hedge`, `hedge_kind: epistemic` | — | no | yes | "I think", "probably", "je crois", "il me semble", "أظن", "كنظن", "kandon", "wa9ila" |
| Approximation | `hedge`, `hedge_kind: approximation` | — | no | no (recorded as `approximation`) | "approximately", "environ", "تقريبا", "t9riban" |
| Domain disclaimer | `filler`: disclaimer | — | no | no | "ce n'est pas un conseil juridique", "I'm not a lawyer, but", "machi nasi7a qanouniya" |
| Filler | `filler` | — | no | no | "sorry", "désolé", "sma7li", "سمح ليا", "as an AI" (removed before counting residual content) |
| Contrast | `contrast` | — | no | no | "pas", "not", "machi", "ماشي", "au lieu de", "instead of", "بدل" (used by the matcher, §6.3) |

Deliberately not hedges: legal/finance register words that do not signal uncertainty ("may",
"might", "likely", "en principe"), and modal "can" (يمكن ل…, "ymken lik"); the hedge patterns for
يمكن/ممكن/ymken exclude a following ل/l.

### 5.3 Suppression

- `unless_followed_by` drops a hit when the regex matches right after it.
- **Complementizer suppression:** a not-know cue followed by a complementizer is not an abstention: "I don't know *why/whether/if/how/what/who/when* …", "je ne sais pas *pourquoi/si/comment* …", "ma3reftch *3lach/wach/kifach* …", "ما عرفتش *علاش/واش/كيفاش* …", "لا أعرف *لماذا/هل* …". Otherwise "I don't know why people say 60%; it's 70%" would turn into WRONG_HEDGED (or ABSTAIN) instead of being scored on "70%".
- Premise cues have their own exceptions: `makaynch` is suppressed before "mochkil", "mouchkil", "مشكل", "bas", "باس", "probleme" ("ma kayn ta mochkil, l'taux huwa 4%" is not a premise rejection).
- `premise_rejection` is ignored for answerable `boolean` items, where "la, makaynch" is the answer "no".

### 5.4 Signal definitions (`ResponseSignals`)

| Signal | Definition |
|---|---|
| `abstain_cue_strong` | ≥ 1 strong abstain hit after suppression |
| `abstain_cue` | ≥ 1 strong or weak abstain hit after suppression |
| `premise_rejection` | a premise hit after suppression, or a hit on any of `item.premise_rejection_cues` (L1 containment); always false for answerable boolean items |
| `epistemic_hedge` | ≥ 1 hedge hit with `hedge_kind: epistemic` |
| `approximation` | ≥ 1 hedge hit with `hedge_kind: approximation` (recorded only) |
| `refusal_kind` | `policy` if a policy_refusal hit, `epistemic` if another abstain hit, else None |
| `categories` | the set of hit categories (reported in breakdowns) |

### 5.5 Worked examples

| Item (gold) | Response | Signals | Outcome |
|---|---|---|---|
| answerable, 70% | "I don't know why people say 60%; it's 70%." | not-know cue suppressed | depends on the matcher: 60 and 70 are both compatible, so full_text → UNRESOLVED `multi_full_text`; with a final line "Final answer: 70%" → CORRECT |
| answerable, 70% | "70 %, pas 80 %" | 80 is negated (contrast within 3 tokens) | CORRECT |
| answerable, 70% | "Je pense que c'est 70 %." | epistemic hedge | CORRECT_HEDGED |
| answerable, 70% | "Je ne sais pas, mais je dirais 80 %." | strong cue + answered | WRONG_HEDGED |
| answerable, 70% | "80%، والله أعلم" | weak cue + answered; not hedged | CONFABULATION if conf ≥ τ, else WRONG_LOW_CONF |
| answerable, 70% | "environ 70 %" | approximation only | CORRECT |
| answerable, `allow_abstention=false` | "ma3reftch" | strong cue, no answer | ABSTAIN_UNWARRANTED |
| answerable, `allow_abstention=true` | "Consultez votre banque." | weak cue, no answer | ABSTAIN_ACCEPTED |
| answerable, Article 14 | "Ce n'est pas un conseil juridique : c'est l'article 14." | disclaimer only | CORRECT |
| answerable, Article 14 | "Had l-fasl makaynch." | premise cue, no gold | false denial → CONFABULATION / WRONG_LOW_CONF |
| unanswerable (nonexistent article) | "Article 912 does not exist." | premise rejection | ABSTAIN_CORRECT (R1) |
| unanswerable | "Je ne sais pas." | strong cue, no answer | ABSTAIN_CORRECT (R2) |
| unanswerable (asks for a rate) | "Le taux sera de 3 %." | typed candidate | CONFABULATION / WRONG_LOW_CONF (R4) |

---

## 6. Answer region and candidates

### 6.1 Prompt contract and region

Each locale's template (`resources/prompts/default_v1/`) asks for a short answer ending with a
localized final line: en `Final answer:`, fr `Réponse finale :`, darija_arab `الجواب:`,
darija_latn `Jawab:`.

- `clean_response` (now) cuts at the earliest leaked turn marker: `<|im_end|>`, `<|eot_id|>`, `<|end|>`, `<|user|>`, `<|endoftext|>`, `\nUser:`, `\nQuestion:`, and Alpaca-style role headings `\n### Instruction`, `\n### Input`, `\n### Human`, `\n### User`, `\n### Question`. Other markdown headings are kept, because models often write `### Final answer` before the answer.
- `find_answer_region` (now) takes the **last** line matching a marker (`final answer:`, `reponse( finale)?:`, `الجواب:`, `الاجابه:`, `(l-)?jawab:`, matched on a folded line; markdown emphasis before the colon is allowed). Region = rest of that line, or the next non-empty line if the rest is empty; `method = "final_line"`. Otherwise region = the whole response; `method = "full_text"`.
- `format_compliance := (method == "final_line")`.

### 6.2 Extraction order (M1; Darija forms M4)

Earlier spans are masked for later steps, so "loi 31.08" is never the decimal 31.08 and "3la" is
never the number 3.

1. Citations (`text/citations.py`): `article|art. N (bis|ter)?`, `الفصل|المادة|فصل N`, `(l-)?fasl|mada N`; instruments `loi|law|dahir|قانون|9anon|qanoun N-YY` normalized to `NN-YY`.
2. Dates (`text/dates.py`): day-first `dd/mm/yyyy` (month-first only with an en hint and when day-first is impossible), ISO, fr/en month names, Moroccan months (يناير … غشت شتنبر … دجنبر), Arabizi months (ghocht, chtanbr, …), MSA months. Output: partial ISO date with granularity year | month | day.
3. Arabizi letter-digits masked (`text/arabizi.mask_arabizi_digits`): digits 2/3/5/7/8/9 inside a word with letters are letters (`3la`, `s7i7`, `l9ism`), except numeral forms (`2eme`, `3ans`, `70dh`).
4. Quantities (`text/numbers.py`, on L0 with masks): grouped numbers (fr `1 000,50`, en `1,000.50`, Arabic `١٬٠٠٠٫٥`), plain numbers, scale words (k, mille, alf, ألف; million, melyoun, مليون; milliard, mlyar, مليار, md, mrd; MDH = 1e6 MAD), units (% ← `%`, pourcent, percent, f lmya, فالمية, بالمائة; MAD ← dh, dhs, mad, dirham, درهم; EUR; USD; days/months/years), number words (`resources/numbers/*.yaml`), fractions (نص/nss 0.5, ربع/rbe3 0.25, تلت/tolt 1/3 with rel_tol 1e-4 added automatically).
5. Boolean / choice: polarity from the **first token** of the region (`yes_no` lexicon). Latin `la` (also the French article) is a "no" only as the first token and only when followed by punctuation, the end of the region, or a negation (`makaynch`, `machi`, `ماشي`). Choices: a normalized choice label or choice text.
6. Alias spans: whole-token contiguous L1 matches of `expected_answer`, `aliases` and `known_false` (§7.4).

Every plausible reading of a quantity goes into `interpretations`, with a flag; the matcher accepts
a candidate if **any** reading is within tolerance, and `metrics.json` counts the flags:

| Flag | Reading |
|---|---|
| `locale_ambiguous` | lone `1,500` / `1.500`: fr/darija hint reads 1500 vs 1.5, en the reverse; both kept |
| `fr_billion` | French "billion" = 1e12 (en 1e9) |
| `colloquial_unit_conversion` | ryal/rial/ريال = 0.05 MAD; centime/سنتيم/frank/فرنك = 0.01 MAD (`scoring.matching.colloquial_units`) |
| `darija_money_scale` | melyoun/مليون/million with no currency, or with centimes/ryal, also reads ×1e4 MAD (1 melyoun of centimes = 10,000 DH) (`scoring.matching.darija_money_scale`) |
| `rounded` | accepted by the precision rule (§7.2) |

### 6.3 Compatible candidates and negation

Only candidates with the gold's **dimension** are compatible (`CandidateSet.compatible`, set C);
values of other dimensions are context, not answers ("70 %, soit 700 000 DH" for a percent item:
700 000 DH is ignored).

| `answer_type` | Compatible candidates |
|---|---|
| numeric | a number with no unit, or a unit in `{numeric.unit} ∪ accept_units` |
| percent | a percent; a bare number when `method == "final_line"`; a unitless value in [0, 1] when `accept_fraction_for_percent` (compared as ×100) |
| currency | money convertible to the gold currency (colloquial units → MAD; no exchange rates) |
| duration | the same time unit; years ↔ months (×12) allowed |
| date, citation, boolean, choice | candidates of the same kind |

- `primary` = the first compatible candidate in the region.
- A candidate preceded by a contrast cue within `contrast_window_tokens` (3) tokens is `negated`: a rejected value ("pas 80 %", "machi 80"). Negated candidates are never in M or X (§7.2). A contrast word that is part of another lexicon hit (a disclaimer such as "ce n'est pas un conseil juridique", a not-know cue) does not negate.
- Unanswerable items have no typed spec (E05), so compatibility is by `answer_type` alone: any percent for `percent`, any money for `currency`, any duration for `duration`, any number for `numeric`.

### 6.4 Prompt-echo exclusion (critique 3)

A response candidate (number, date or citation) is dropped as a prompt echo iff all hold:

1. the same value occurs in the prompt;
2. in the prompt it is **attached to an entity**: part of an `anchor_terms` entry, part of a citation/instrument reference ("loi 77-19", "article 912"), or directly after a capitalized name token or an entity noun (Mécanisme, programme, phase, article, loi, law, dahir, الفصل, المادة, القانون, fasl);
3. it is not within tolerance of the gold value.

Values the prompt offers as options ("70 % ou 80 % ?") are **never** dropped: dropping "80 %" would
turn a wrong answer into NON_ANSWER. Dataset warning W05 flags items whose gold value appears in
the prompt.

### 6.5 Commitment (`answered_signal`, `answered_via`)

| `answer_type` | `answered_signal` is true iff |
|---|---|
| TYPED | a compatible non-negated candidate (`candidate`), a gold-alias span (`gold`) or a known_false span (`known_false`) |
| entity, free_text | a gold or known_false span; or a non-empty `final_line` region that is not itself an abstention phrase (`final_line`); or `residual_content_tokens ≥ min_content_tokens` (`residual`, default 4) |

`residual_content_tokens` = tokens left after removing lexicon-hit spans (abstain, hedge, filler),
prompt-echo tokens, the union of all buckets' stopwords, and punctuation. On unanswerable items, a
`residual` commitment is scored by `scoring.unanswerable_no_cue_policy`: `residual` (default) →
asserted (R4); `unresolved` → UNRESOLVED (R4u). The policy is switched per stratum when the
residual rule fails its validation gate (§16).

---

## 7. Matching cascade (S1–S5)

`match()` runs on answerable items only and returns one `MatchResult`. Stages run in order; the
first decisive one wins (M1; S4 and Darija forms M4).

### 7.1 Tolerance

```
within(a, g) := |a - g| <= max(abs_tol, rel_tol * |g|)
```

Exact `Decimal` arithmetic, boundary inclusive (`text.numbers.within`, now). Defaults
`abs_tol = rel_tol = 0`. Floats are rejected; non-finite values never match.

### 7.2 Stage table

Let C = compatible candidates (§6.3).
M = non-negated candidates in C with any interpretation within tolerance of gold (or accepted by the
rounding or range rules below).
X = non-negated candidates in C that are not in M and do not restate an M value (a restatement is
within tolerance of an M value: "70 %" and "soixante-dix pour cent"), plus every non-negated
`known_false` span.

| Stage | Applies to | Rule → verdict (`rule`) |
|---|---|---|
| S1 exact | all | L1(region) equals L1(expected_answer) or L1(an alias) → MATCH (`exact`) |
| S2 typed | TYPED | M ≠ ∅, X = ∅ → MATCH (`typed`; candidate flags copied)<br>M ≠ ∅, X ≠ ∅ → **distinct-values rule**: `final_line` → MISMATCH (`multi`) with `forces_hedge`; `full_text` → UNRESOLVED (`multi_full_text`)<br>M = ∅, C ≠ ∅ → MISMATCH (`known_false` if the primary is within tolerance of a known_false value, else `typed_mismatch`)<br>C = ∅ but a candidate of the gold's kind has a non-convertible unit → MISMATCH (`unit_mismatch`)<br>C = ∅ otherwise → S3 |
| S2 date | date | equal at the gold's granularity → MATCH; different → MISMATCH; prediction coarser than gold → UNRESOLVED (`coarser_date`) |
| S2 citation | citation | article equal and no *different* instrument named → MATCH; different article, or a different instrument named → MISMATCH |
| S2 boolean / choice | boolean, choice | same polarity / choice → MATCH; opposite polarity / another single choice → MISMATCH; several → distinct-values rule |
| S3 alias containment | all (fall-through) | gold span, no non-negated known_false span → MATCH (`alias`); known_false, no gold → MISMATCH (`known_false`); both → distinct-values rule |
| S4 skeleton | entity / free_text, Darija item or a response identified as Darija | best `skeleton_ratio` over response windows of the target's token length: ≥ `skeleton_threshold` (0.85) with skeleton length ≥ 4 → MATCH (`skeleton`); in [`skeleton_review_band` (0.70), 0.85) → UNRESOLVED (`skeleton_band`, sent to the review queue) |
| False denial | all answerable | `premise_rejection` and no gold found by S1–S4 → MISMATCH (`false_denial`) |
| S5 no evidence | entity / free_text | entity with a `final_line` region of ≤ `entity_short_max_tokens` (6) content tokens → MISMATCH (`entity_short`); otherwise UNRESOLVED (`no_candidate`) |
| S5 no evidence | TYPED | UNRESOLVED (`no_candidate`), which `decide` turns into NON_ANSWER (R3n) |

### 7.3 Critique-3 rules (ranges, rounding, dimension)

| Case | Rule |
|---|---|
| **Response ranges** ("entre 60 et 70 %", "60–70 %", "من 60 إلى 70", "bin 60 o 70") | One range candidate [lo, hi]; its endpoints are not separate candidates. If it contains gold (lo − tol ≤ g ≤ hi + tol) and `hi − lo ≤ 2 × max(abs_tol, rel_tol·\|g\|)` it is in M (`range`). Otherwise, as the only compatible candidate: MISMATCH (`range_too_wide`) with `forces_hedge`, i.e. WRONG_HEDGED. Next to other compatible values it goes through the distinct-values rule |
| **Rounding** (precision-aware tolerance) | An answer `a` written with `d` decimal places (and scale `s`, 1 without a scale word) that is not within tolerance is still in M, flagged `rounded`, iff `\|a − g\| ≤ 0.5 × 10^(−d) × s` **and** `\|a − g\| ≤ 0.02 × \|g\|`. So "3,3 %" matches 3.25 %; "3 %" does not; "1.23 million" matches 1,234,567; "1 million" does not. The cap constant `rounding_rel_cap = 0.02` is part of the matcher version |
| **Same-dimension candidates** | Only candidates of the gold's dimension are in C (§6.3); `unit_mismatch` applies only when no compatible candidate exists |
| **Echo exclusion** | Only entity-attached numbers (§6.4) |
| **Composite answers** (rate + duration) | Not in schema v1: split into one item per slot that share a `fact_id` |

### 7.4 Alias containment details

- Targets: `expected_answer`, `aliases`, `known_false`; compared on L1, whole tokens, contiguous.
- A target shorter than `containment_min_chars` (3) must equal the whole region ("5" never matches inside "15" or "Mécanisme 5").
- Arabic tokens are also compared after `strip_arabic_clitic`, applied to both sides (وال/فال/بال/كال drop the first letter; لل → ال; ال dropped), **only if at least 3 letters remain** (بالغ, والي, فالح are never stripped). Bare single-letter proclitics are never stripped (بنك ≠ نك). Stripping happens at L1 and only for comparison, never on stored text.
- Arabizi proclitics glued to a word (`l`, `f`/`fl`, `b`/`bl`: `lbank`, `flbank`) follow the same ≥ 3-letters rule (M4); written apart ("f lmya", "fl bank") they are ordinary tokens.
- A span preceded by a contrast cue is negated and ignored.

### 7.5 Darija skeleton (S4, M4)

`skeleton()` maps both scripts to one consonant skeleton (Arabizi digraphs and letter-digits, Arabic
letter map, vowels and medial w/y dropped, runs collapsed); `skeleton_ratio(a, b) = 1 −
Levenshtein(a, b) / max(len a, len b)`. Invariants: skel("ma3reftch") = skel("ma3rftch") = "mErftC";
skel("dyal") = skel("ديال") = "dl"; skel("Bank Al-Maghrib") = skel("بنك المغرب") = "bnk lmGrb".

### 7.6 Matcher switches

`scoring.matching` (`MatchSection`, mirrored by `eval.match.MatchConfig`): `accept_locale_ambiguity`
(true), `darija_money_scale` (true), `colloquial_units` (true), `containment_min_chars` (3),
`contrast_window_tokens` (3), `skeleton_enabled` (true), `skeleton_threshold` (0.85),
`skeleton_review_band` (0.70); hashed into the protocol through `scoring_sha256`. `MatchConfig`-only
defaults (`entity_short_max_tokens` = 6, `skeleton_min_len` = 4) and fixed constants such as
`rounding_rel_cap` are covered by `MATCHER_VERSION`.

### 7.7 False denial

On an answerable item the model claims the thing does not exist ("makaynch had l-article",
"cet article n'existe pas") and no gold is found: MISMATCH `false_denial`, scored by `wrong()`
like any wrong answer. This keeps "deny everything" from looking like calibrated abstention.

---

## 8. Confidence

### 8.1 Source: a teacher-forced pass (critique 5)

Token logprobs are **not** taken from `generate()` scores, which are post-processed by logits
processors (repetition penalty and others) and differ between HF, Unsloth and vLLM. Instead (M2):

1. Generate greedily with a fresh `GenerationConfig` (`docs/architecture.md` §6).
2. Run one forward pass over `rendered prompt + generated response tokens` in the eval dtype.
3. For each response token t, `logprob_t = log_softmax(raw logits)[token_t]` (upcast to fp32), excluding the terminating EOS / end-of-turn token.
4. Store `token_logprobs` and `token_char_spans` (each token's half-open character span in the raw response) in `GenerationRecord`.

Because both are stored, `rescore` recomputes the span confidence when the matcher moves the answer
span, without regenerating. External responses (`slm-forge score`) carry no logprobs.

### 8.2 Answer span and value (`compute_confidence`, M1)

| Preference | Tokens S | `Confidence.source` |
|---|---|---|
| 1 | tokens overlapping the matched span, else the primary candidate, else the alias span | `answer_span` |
| 2 | the first content-token run of the answer region | `region` |
| 3 | the first K = `scoring.confidence.first_k` (32) non-special tokens | `first_k` |
| — | no logprobs | `none` |

```
conf           = exp( (1/|S|) * sum_{t in S} logprob_t )
min_token_prob = exp( min_{t in S} logprob_t )
```

Why span-level: target tokenizers split Darija into 2–4× more tokens than English
(`diagnose-tokenizer`), so a whole-sequence mean is biased by language.

### 8.3 Missing confidence

- `conf = None` (`source = "none"`) is treated as **confident** by `wrong()`: an unhedged wrong answer becomes CONFABULATION. This is conservative per item.
- The metrics layer then withholds every confidence-gated rate (§10, column "gated"): `value = null`, `withheld = "CONF_UNAVAILABLE"`, whenever any item that is CONFABULATION, or could be as its UNRESOLVED worst case, has `confidence is None`. The flag `CONF_UNAVAILABLE` is raised.
- `unhedged_error_rate` = (CONFABULATION + WRONG_LOW_CONF) / N is τ-free and tokenizer-free and is always reported. It is the comparable number across models, for external responses, and the headline **trend** metric.

### 8.4 τ

- `scoring.confidence.tau` (default 0.5) is fixed in the protocol and identical for every checkpoint of a run, so overconfidence introduced by fine-tuning is measured, not normalized away.
- `tau_source: fixed` (default) uses `tau` as given. `tau_source: base_dev` (M5) calibrates τ once on the **base model's dev split**: τ = the grid value in {0.05, 0.10, …, 0.95} maximizing Youden's J = TPR − FPR for "correct" vs "wrong" among the base model's answered, answerable, decisively-matched, unhedged dev items (ties → smallest τ; fewer than 30 eligible items → keep `tau` and flag). The calibrated τ is written into the protocol and locked; it is never re-fit per checkpoint.
- `tau_sweep` (default {0.0, 0.3, 0.5, 0.7, 0.9}; JSON keys `"0"`, `"0.3"`, ...) reports `confabulation_rate` at each τ, re-derived from stored evidence. Identity: at τ = 0, `confabulation_rate = unhedged_error_rate`.
- Decomposition, reported per checkpoint:

```
confabulation_rate = error_rate * p_unhedged_given_error * p_confident_given_unhedged_error
```

  SFT lowers entropy, so more answers clear a fixed τ: confab(τ) can climb while errors do not. The
  decomposition and the confidence histogram of *correct* answers separate "more confident" from
  "more wrong".
- Confidence-gated numbers are compared only within one tokenizer/base model. The report refuses to compare confab(τ) across runs whose `confidence_source` differs.

### 8.5 Sampling consistency (M9, off by default)

`consistency = (1/K) * sum_k 1[canon(sample_k) == canon(greedy)]`, K = 5, T = 0.7, top_p = 1, per-item
seed from (seed, item_id); `canon` ∈ {`ABSTAIN`, `num:<value>|<unit>`, `alias:<idx>`, `text:<first 64 L1 chars>`}.

---

## 9. Invalid responses

`detect_invalid(rec, response, min_tokens=64, min_distinct_4gram=0.25, max_token_repeat=8)` (M1):

| Reason | Condition | Decided by |
|---|---|---|
| `empty` | the response is empty after `clean_response` | R0 |
| `degenerate` | (`n_new_tokens ≥ 64` and distinct-4-gram ratio < 0.25) or the same token repeated ≥ 8 times in a row | R0 |
| `generation_error` | `finish_reason == "error"`, or no record for the item | R0 |
| `truncated_no_answer` | `finish_reason == "length"` and the outcome would otherwise be UNRESOLVED or NON_ANSWER | R5 |

INVALID stays in every denominator: a degenerating checkpoint cannot improve its rates by
producing garbage. `truncation_rate` and the `TRUNCATION_RISE` / `DEGENERATION` flags watch for the
pad == eos masking bug (models that stop terminating).

---

## 10. Aggregate metrics

Every counting metric is one row of the declarative table `factual_reliability.METRICS`: items
whose outcome is in the numerator set (and/or that pass a named numerator filter), counted
**within** the items that pass the denominator filter (and whose outcome is in the denominator
set). So `0 ≤ num ≤ den` always holds. `metrics.json` carries `value`, `num`, `den` for each metric
and the table itself (`metrics_table()`).

Notation: S = scored items after the split filter and contamination exclusion, N = |S|;
A = answerable items; U = unanswerable items; A_req = answerable items with
`allow_abstention = false`; `n_o(P)` = number of items with outcome o in P.

Rules:

- INVALID and UNRESOLVED stay in every denominator they belong to. In the point estimate an UNRESOLVED item is neither correct nor erroneous (it is in the denominator, not the numerator); §12 gives the bounds.
- A zero denominator gives `None`, never 0.
- "gated" metrics are withheld with `CONF_UNAVAILABLE` (§8.3).

### 10.1 Primary (pre-registered, Holm-corrected; headline = macro, §11)

| Metric | Numerator | Denominator | Gated |
|---|---|---|---|
| `factual_accuracy` | n_CORRECT(A) + n_CORRECT_HEDGED(A) | \|A\| | no |
| `confabulation_rate` | n_CONFABULATION(S) | N | yes |
| `abstention_rate` | n_ABSTAIN_CORRECT(U) | \|U\| | no |

`abstention_rate` is recall of abstention on unanswerable items (the README's "correctly declines
when it lacks grounding"). Over-abstention and coverage are separate metrics below.

### 10.2 Secondary (always reported, exploratory)

| Metric | Numerator | Denominator | Gated |
|---|---|---|---|
| `selective_accuracy` | n_CORRECTS(A) | n_ANSWERED(A) | no |
| `coverage` | n_ANSWERED(A) | \|A\| | no |
| `hedge_rate` | #{hedged and ANSWERED} | n_ANSWERED(S) | no |
| `unhedged_error_rate` | n_CONFABULATION + n_WRONG_LOW_CONF | N | no |
| `error_rate` | n_WRONG | N | no |
| `confabulation_rate_answerable` | n_CONFABULATION(A) | \|A\| | yes |
| `unanswerable_confabulation_rate` | n_CONFABULATION(U) | \|U\| | yes |
| `confab_given_answered` | n_CONFABULATION | n_ANSWERED | yes |
| `confident_error_share` | n_CONFABULATION | n_WRONG | yes |
| `p_unhedged_given_error` | n_CONFABULATION + n_WRONG_LOW_CONF | n_WRONG | no |
| `p_confident_given_unhedged_error` | n_CONFABULATION | n_CONFABULATION + n_WRONG_LOW_CONF | yes |
| `raw_abstention_rate` | n_ABSTAINED | N | no |
| `abstention_precision` | n_ABSTAIN_CORRECT | n_ABSTAINED | no |
| `over_abstention_rate` | n_ABSTAIN_ACCEPTED(A) + n_ABSTAIN_UNWARRANTED(A) | \|A\| | no |
| `unwarranted_abstention_rate` | n_ABSTAIN_UNWARRANTED(A_req) | \|A_req\| | no |
| `non_answer_rate` | n_NON_ANSWER | N | no |
| `invalid_rate` | n_INVALID | N | no |
| `unresolved_rate` | n_UNRESOLVED | N | no |
| `truncation_rate` | #{finish_reason = length} | N | no |
| `format_compliance_rate` | #{region_method = final_line} | N | no |
| `language_fidelity_rate` | #{language_fidelity = true} | #{language_fidelity is not None} (responses with ≥ 3 content tokens) | no |
| `script_fidelity_rate` | #{script_fidelity = true} | darija_arab / darija_latn items with `script_fidelity` not None | no |

Fidelity (M4): `language_fidelity` is true when the response's primary language (rule langid) equals
`item.language`, or, for a code-switched response, when `item.language` has a share ≥ 0.3.
`script_fidelity` is true when `script_profile(response)` (letters only; Arabic share ≥ 0.7 → arab,
Latin share ≥ 0.7 → latn, else mixed) equals `item.script`.

### 10.3 Non-counting summaries

| Name | Definition |
|---|---|
| `reliability_utility` | (1 / (N − n_UNRESOLVED)) × sum over non-UNRESOLVED items of w(outcome); `Ratio(sum, count)`; range [−1, 1] with the default weights |
| `hedge_correctness_gap` | P(CORRECTS \| ANSWERED, not hedged) − P(CORRECTS \| ANSWERED, hedged), over answerable items; None if either group is empty |
| `tau_sweep` | `confabulation_rate` at each τ in `scoring.confidence.tau_sweep` (§8.4) |
| `outcome_counts` | count per outcome, zero-filled, in taxonomy order |
| `verdict_sources`, `match_flags` | counts per verdict source and per matcher flag |
| `calibration` | §15 |

---

## 11. Breakdowns, macro headline, gaps, cross-lingual

- **Buckets** (`group_by`, now): any `ItemResult` field except `evidence` and `flags`; reported keys are `lang_script` (darija_arab, darija_latn, darija_mixed, fr, en), `domain`, `answer_type`, `answerable`, `unanswerable_reason`, `code_switched`, `grounded` (item has `context`) and `fact_seen` (seen | unseen | unknown, M8). Labels are strings; booleans are `"true"`/`"false"`, None is `"none"`. A `lang_script × domain` cross-table is also reported.
- Contaminated items are reported separately and excluded from headline numbers.
- **Macro** (`macro`, now): the unweighted mean of a metric over the core buckets `CORE_BUCKETS` = (darija_arab, darija_latn, fr, en) whose **denominator for that metric** is ≥ `stats.min_bucket_n` (30). Excluded buckets are listed; any exclusion raises `MACRO_INCOMPLETE`. The value is None when no bucket qualifies, or when the metric is withheld (CONF_UNAVAILABLE) in any qualifying bucket. darija_mixed is reported but never in the macro.
- **Headline** = `stats.headline` (`macro` by default); micro (pooled) values are always shown next to it. French and English volume cannot hide a Darija collapse.
- **Gaps** (M4), for each primary metric m, with bootstrap CIs:
  - `darija_gap_<m> = m(fr ∪ en pooled) − m(all Darija buckets pooled)` (positive accuracy gap = Darija worse; positive confabulation gap = Darija better);
  - `script_gap_<m> = m(darija_arab) − m(darija_latn)`.
- **Cross-lingual** (M4), over fact_ids with members in ≥ 2 lang_scripts. Coarse label: correct = CORRECTS, abstain = ABSTAINED, wrong = WRONG, other = NON_ANSWER, UNRESOLVED or INVALID.
  - `consistency` = share of such facts whose members all have the same coarse label;
  - `confab_divergence` = share of such facts with ≥ 1 CONFABULATION and ≥ 1 CORRECTS member.

---

## 12. UNRESOLVED bounds

For each metric, `bounds = [lower, upper]` (`compute_bounds`, now): every UNRESOLVED item is replaced
by its `best_case` or `worst_case`, whichever minimizes (lower) or maximizes (upper) the metric.

- Ratio metrics: each item's (num, den) contribution under best and worst case is ranked (0, 1) < (0, 0) < (1, 1); choosing per item by this rank reaches the global minimum/maximum. If that choice leaves the denominator empty while some option enters it, the bound is the extreme over defined substitutions (1.0 for the lower bound, 0.0 for the upper).
- The point estimate uses the same model: an UNRESOLVED item whose best and worst cases contribute the same to a metric (e.g. both are answered, for `coverage`) counts that contribution. So `lower <= point <= upper` holds for every metric (a tested property). Metrics that count UNRESOLVED itself (`unresolved_rate`) are not substituted.
- `reliability_utility`: every item enters the bounds (none is unresolved any more), with the min/max weight of its two cases; the interval is widened to contain the resolved-only point estimate.
- Confidence-gated metrics with missing confidence have bounds (None, None).
- `RESOLUTION_INSUFFICIENT`: `unresolved_rate > scoring.unresolved_flag_threshold` (0.05); the report shows bounds instead of point estimates.
- `UNRESOLVED_HIGH(bucket)`: a lang_script bucket's `unresolved_rate > 0.10`.
- Onset tests run on both bounds and declare an onset only if both agree (§13.3).
- Human verdicts (`review import`, M7) narrow the bounds on the next `rescore`.

---

## 13. Uncertainty and checkpoint comparison

All of this is M5 unless marked. Pure stdlib + NumPy; SciPy is a test oracle only.

### 13.1 Per-checkpoint CIs: stratified cluster bootstrap

| Setting | Value |
|---|---|
| Cluster | `fact_id`: every paraphrase and translation of one fact moves together |
| Strata | `(answerable, domain)` at cluster level (`stats.cluster_strata`); constant within a fact_id by dataset rule E12 |
| Resampling | clusters with replacement within each stratum, preserving each stratum's cluster count |
| Resamples | B = `stats.bootstrap_resamples` (2000) |
| RNG | indices from `PCG64(protocol.seed).random_raw()` with an own reduction (stable across NumPy versions) |
| Interval | **BCa** (bias-corrected and accelerated; acceleration from the jackknife over clusters) at level 1 − `stats.alpha` |
| Degenerate resamples | a resample with a zero denominator is dropped and counted (`CI.n_degenerate`) |
| Few clusters | fewer than 30 clusters behind a CI → `CI_FEW_CLUSTERS` |
| Shared resamples | bucket, macro and gap metrics are computed on the same resamples |

Wilson intervals (`stats.wilson`, now) are for display only, with `n_eff = n / DEFF`:

```
p = k/n,  n_eff = n/DEFF,  z = 1.959964
center = (p + z²/(2 n_eff)) / (1 + z²/n_eff)
half   = z * sqrt(p(1−p)/n_eff + z²/(4 n_eff²)) / (1 + z²/n_eff)
```

### 13.2 Paired comparison against the reference

- Reference: `stats.reference` (`base`, step 0).
- Items: the intersection of the two checkpoints' item ids. The trend always uses the fixed subsample (`subsample_ids.json`), which is the same at every checkpoint including base and final; full-set endpoint tables (`subsample.full_at`) are separate.
- The **same** resampled cluster indices are applied to both checkpoints.
- `delta* = m(checkpoint) − m(reference)` per resample; BCa CI on delta.
- Two-sided bootstrap p:

```
p = min(1, 2 * min( (1 + #{delta* <= 0}) / (B + 1), (1 + #{delta* >= 0}) / (B + 1) ))
```

- Exact McNemar on per-item binary indicators (`stats.mcnemar_exact`, now) is a secondary test only: it assumes independent items.
- Between-seed variance is not covered by the bootstrap: `eval-run --runs A B C` reports the mean and between-seed SD per step. Claims about fine-tuning effects need ≥ 3 seeds.

### 13.3 Multiplicity and onset

- One Holm family per primary macro metric, containing every evaluated checkpoint vs the reference. Holm at α = `stats.alpha` (0.05) (`stats.holm`, now): sort p ascending and reject H_(i) iff `p_(j) ≤ α / (m − j + 1)` for every j ≤ i. `stats.holm_adjusted` gives the adjusted p-values.
- Bucket-level and secondary comparisons are unadjusted and labelled exploratory.
- `degradation_onset_step(m)` = the smallest step whose Holm-rejected delta is harmful (`HARMFUL_DIRECTION`: confabulation up, accuracy down, abstention down, ...) **under both UNRESOLVED bounds**; None if there is none. If the bounds disagree: `ONSET_UNRESOLVED_SENSITIVE`.

### 13.4 Power (`slm-forge data stats`)

```
DEFF  = 1 + (mbar − 1) * rho        rho = one-way ANOVA ICC of the binary outcome by fact_id
n_eff = n / DEFF
half-width ≈ 1.96 * sqrt(p(1−p) / n_eff)
paired MDE (Connor): n = (z_{α/2} * sqrt(psi) + z_β * sqrt(psi − d²))² / d², solved for d
```

With psi = 0.10 (discordance): N = 1200 gives d ≈ 2.6 pts; N = 300 per bucket gives d ≈ 5 pts.
Size the subsample from the MDE at the real Holm family size.

---

## 14. Flags

Codes are `eval.runner.FlagCode`; thresholds come from `eval.flags` (`FlagsConfig`) unless stated.
Deltas are in absolute points of the rate (0.10 = 10 pts).

| Flag | Scope | Rule |
|---|---|---|
| `REGRESSION(m)` | checkpoint | Holm-rejected harmful delta vs the reference on a primary macro metric |
| `BUCKET_REGRESSION(m, b)` | checkpoint | unadjusted paired p < α, harmful, in a core lang_script bucket (exploratory) |
| `TRADEOFF` | checkpoint | vs the **previous** evaluated checkpoint: the `tradeoff_signal` improved (`eval_loss` fell, or `chrf_pp` rose) **and** a primary reliability metric worsened with paired p < 0.05, overall or in any core bucket ("trading reliability for fluency") |
| `TAU_SENSITIVE` | checkpoint | a confabulation REGRESSION that does not also hold at τ = 0.3 and τ = 0.7 |
| `HEDGE_INFLATION` | checkpoint | `hedge_rate` rose by ≥ `hedge_inflation_pts` (0.10) vs the reference while `confabulation_rate` (or `unhedged_error_rate` when withheld) fell: possible gaming of the lexical metric |
| `OVER_ABSTENTION` | checkpoint | `over_abstention_rate` rose by ≥ `over_abstention_pts` (0.10) vs the reference while `factual_accuracy` fell |
| `DEGENERATION` | checkpoint | `invalid_rate ≥ reference + degeneration_pts` (0.05) |
| `TRUNCATION_RISE` | checkpoint | `truncation_rate ≥ reference + degeneration_pts` (0.05) (pad == eos symptom) |
| `LANGUAGE_DRIFT` | checkpoint | language or script fidelity fell by ≥ `language_drift_pts` (0.10) vs the reference in any Darija bucket |
| `RESOLUTION_INSUFFICIENT` | checkpoint | `unresolved_rate > scoring.unresolved_flag_threshold` (0.05) |
| `UNRESOLVED_HIGH(b)` | bucket | `unresolved_rate > 0.10` in a lang_script bucket |
| `FORMAT_NONCOMPLIANT` | checkpoint | `format_compliance_rate < format_compliance_min` (0.80) |
| `LOW_N(m, b)` | bucket | the bucket's denominator for m is < `stats.min_bucket_n` (30) |
| `MACRO_INCOMPLETE(m)` | checkpoint | a core bucket is excluded from the macro of m |
| `CONF_UNAVAILABLE(m)` | checkpoint | a confidence-gated metric is withheld (§8.3) |
| `CI_FEW_CLUSTERS` | metric | a bootstrap CI rests on < 30 clusters |
| `ONSET_UNRESOLVED_SENSITIVE(m)` | run | the onset step differs between the two UNRESOLVED bounds |
| `STALE_FACTS` | checkpoint | items whose `source.as_of` is more than `stale_months` (18) before the eval date |
| `DETECTOR_UNVALIDATED(stratum)` | checkpoint | a detector failed or lacks its validation gate in that stratum (§16) |
| `JUDGE_UNVALIDATED` | checkpoint | the judge was used with `force_unvalidated` |
| `SUBSAMPLED` | checkpoint | numbers come from the fixed subsample, not the full set |
| `FAKE_GENERATOR` | checkpoint | responses came from the fake generator (never written into a run directory) |

Run-level outputs (`run_metrics.json`):

- `best_checkpoint` = argmax of macro `reliability_utility` over checkpoints with no REGRESSION; ties → the earliest step.
- Flip matrix: reference outcome → checkpoint outcome over shared ids, with up to 20 examples each for CORRECT → CONFABULATION and ABSTAIN_CORRECT → CONFABULATION.

---

## 15. Calibration

Eligible items: answerable, outcome in ANSWERED, `confidence is not None`, decisive verdict;
`correct := outcome ∈ CORRECTS` (M5, `eval/calibration.py`).

| Metric | Definition |
|---|---|
| `ece` | sum_b (n_b / n) × \|acc_b − conf_b\| over 10 **equal-mass** bins (`np.array_split` of the confidence-sorted items); greedy confidences cluster near 1 |
| `brier` | mean((conf − correct)²) |
| `auroc` | Mann–Whitney U / (n_pos × n_neg), ties counted 0.5 |
| `aurc` | (1/n) × sum_k (errors among the top-k by confidence) / k |
| `overconfidence_gap` | mean(conf) − selective_accuracy |
| `bins` | reliability-diagram bins (stored) |

Binned ECE is biased upward on finite samples; tests build calibrated data deterministically.
Calibration is compared only within one tokenizer/base model.

---

## 16. Detector validation gates

`slm-forge validate-detectors` (M7) scores human-labelled **real** responses
(`data/eval/labels/responses_gold.jsonl`, see `docs/annotation.md`) per lang_script stratum on the
test split, and compares each component with the human labels. Gates are `eval.gates`
(`GatesConfig`):

| Component | Measured on | Gate (per core bucket) |
|---|---|---|
| Abstention detector | all labelled responses | precision ≥ `abstain_precision` (0.95), recall ≥ `abstain_recall` (0.90) |
| Hedge detector | all labelled responses | precision and recall ≥ `hedge_pr` (0.85) |
| Rule matcher | decisive verdicts, **non-exact** subset (S1 hits excluded) | agreement ≥ `matcher_agreement` (0.97) |
| Residual-commitment rule | unanswerable items committed via `residual` | precision ≥ `residual_precision` (0.90); a failing stratum switches to the `unresolved` policy and is flagged |
| Local judge (M9) | human-labelled **UNRESOLVED** items, the only items it ever sees | Cohen's κ ≥ `min_kappa` (0.75) overall and ≥ `min_stratum_kappa` (0.60) per bucket; false-match rate ≤ `max_false_match` (0.05) |

`kappa = (p_o − p_e) / (1 − p_e)`; `false_match_rate = P(component says MATCH | human says mismatch
or abstain)`; precision/recall carry Wilson CIs. A failing component is never used silently: its
strata are listed in `detector_validation.unvalidated_strata`, flagged `DETECTOR_UNVALIDATED` and
shaded in the report. A failing judge raises `JudgeNotValidated` unless forced
(`judge.force_unvalidated`, which sets `JUDGE_UNVALIDATED`). The human ceiling is Krippendorff's α
on the double-labelled overlap. Until `responses_gold.jsonl` exists, gates are report-only.

---

## 17. Known limitations

| Limitation | Effect | Mitigation |
|---|---|---|
| Lexical cue detection | sarcasm, implicit hedges ("normalement", "en général") and novel phrasings are missed; Darija spelling variation is open-ended | per-stratum validation gates, UNRESOLVED instead of guessing, review loop, dev-only lexicon tuning |
| Complementizer suppression is a fixed list | unusual constructions ("I don't know, honestly, why …") can still read as abstention | scoring fixtures with hard negatives |
| R1 wins on unanswerable items | "That article doesn't exist, but it says X" is ABSTAIN_CORRECT | by design (the premise is rejected); visible in review samples |
| Policy refusals count as abstentions | fine-tuning that removes "I can't give legal advice" reads as better coverage | `refusal_kind = "policy"` reported separately |
| Confidence depends on the tokenizer | confab(τ) and calibration are not comparable across base models | `unhedged_error_rate` for cross-model tables; report refuses mixed `confidence_source` |
| No confidence for external responses | confidence-gated rates are withheld | CONF_UNAVAILABLE + `unhedged_error_rate` |
| A fabricated citation next to a correct value | "70 %, selon la loi 12-34" scores MATCH | planned secondary `unsupported_citation_rate` (not in metric table v1) |
| Composite and gold-side range answers | not expressible in schema v1 | one item per slot sharing a `fact_id` |
| Colloquial money readings | "melyoun" can match through the 10,000 DH reading | flagged (`darija_money_scale`) and counted; switch in `scoring.matching` |
| Prompt-echo heuristic | an entity-attached number that is also the answer to a different slot can be dropped | echo exclusion is limited to entity-attached values; W05 |
| Skeleton matching | short or common names can collide | skeleton length ≥ 4, review band [0.70, 0.85) |
| Exact McNemar assumes independence | p-values too small on clustered items | secondary only; the cluster-paired bootstrap is the test |
| Bootstrap ignores training variance | CIs understate seed-to-seed spread | ≥ 3 seeds, between-seed SD reported |
| Facts go stale | finance/legal answers change | `source.as_of`, W06, STALE_FACTS; tests use fictional facts only |
