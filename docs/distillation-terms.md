# Teacher distillation: terms, data protection and safeguards

`slm-forge distill` asks a third-party "teacher" model (Anthropic or OpenAI API) to generate
**training** data, for example Darija paraphrases of seed prompts. This page explains what you must
check before turning it on, and what slm-forge enforces. It is not legal advice: slm-forge records
your acknowledgment and makes no legal determination.

## 1. Why it is optional

| Point | Detail |
|---|---|
| Not needed by the pipeline | `finetune`, `eval`, `eval-run`, `score` and `rescore` never call `data/distill.py`. Training data can be entirely human-written |
| One-time data preparation | distillation produces a JSONL file of training records; after that it plays no role |
| Off by default | `distill.enabled: false` in the shipped config (`configs/distill.yaml`) |
| Not part of evaluation | teacher output never enters eval ground truth, gold labels or judge labels (§6) |
| Lazy dependencies | the provider SDKs (`distill` extra) are imported only inside the client classes (M9) |

## 2. The opt-ins (`DistillConfig`)

Running requires all of the following. `DistillConfig` validates them when the config is loaded, and
`data.distill.check_gates` checks them again at run time (a config can be built without validation).
Nothing is sent if any gate fails.

| Gate | Fields | Failure |
|---|---|---|
| Enabled | `enabled: true` | `DistillationDisabled` |
| **Opt-in 1: terms reviewed** | `tos_acknowledged: true`, `terms_reviewed_on: YYYY-MM-DD` (not in the future), `acknowledged_by: <name>` | `ToSNotAcknowledged` |
| **Opt-in 2: data leaves the machine** | `allow_send_data: true` | `DataEgressNotAllowed` |

The acknowledgment (`terms_reviewed_on`, `acknowledged_by`) is written into `distill_manifest.json`.
Re-review the terms and update the date whenever you change provider or model, or at least before
each new distillation campaign.

## 3. Provider terms: what to check

Provider terms change. Read the **current** terms of service, usage policies and any
model-specific terms for the exact account type you use (consumer, API, enterprise). slm-forge
does not summarize or interpret them. Questions to answer before setting `tos_acknowledged`:

| Topic | Question |
|---|---|
| Use of outputs | May outputs be used to train or fine-tune another model at all? Under which conditions? |
| "Competing model" clauses | Do the terms restrict building models that compete with the provider's? These clauses can apply to **the model you train and publish**, not only to the generated data |
| Redistribution | May the generated data itself be published or shared? Under which licence? |
| Your inputs | Are prompts retained, logged or used for training by the provider? Can that be turned off for your account? |
| Personal data | Does the provider act as a processor? Where is data processed and stored? |
| Attribution and disclosure | Must generated content or derived models be labelled? |
| Volume | Rate limits and quotas (`rate_limit_rps`, `max_requests` bound slm-forge's side) |

If a question has no clear answer for your use, do not enable distillation.

## 4. Personal data and Moroccan Law 09-08

- Law No. 09-08 on the protection of individuals with regard to the processing of personal data applies to processing in Morocco and is supervised by the CNDP (Commission nationale de contrôle de la protection des données à caractère personnel).
- Sending prompts to a foreign API is a **transfer of data abroad**. If prompts contain personal data, such a transfer may require a prior declaration to, or an authorization from, the CNDP. Other regimes (for example the EU GDPR) may apply as well, depending on whose data it is.
- The safest course is to send no personal data at all: build seed prompts from public, non-personal material (product sheets, statutes, synthetic questions).
- `allow_send_data: true` records that you made this assessment. It does not make a transfer lawful.

## 5. PII redaction

`redact_pii: true` (default) runs `data.distill.redact_pii` on every prompt **before** it is sent
(now). Matching runs on a digit-folded copy, so Arabic-Indic digits are caught too; redaction is
idempotent, and only the kinds of redactions (never the values) are counted in the manifest.

| Kind | Pattern (summary) | Placeholder |
|---|---|---|
| email | `name@domain.tld` | `[EMAIL]` |
| iban | Moroccan IBAN `MA` + 26 digits, plain or in groups of four | `[IBAN]` |
| rib | 24-digit RIB, plain or grouped 3-3-(4×4)-2 | `[RIB]` |
| phone | `+212` / `00212` / `0`, then 5–7 and 8 digits, plain, in pairs or 3-3-3 | `[PHONE]` |
| cin | national ID card: 1–2 capital letters + 5–6 digits | `[CIN]` |

Redaction is pattern-based. It does not catch names, addresses, or free-text descriptions of a
person, so it is a safety net, not a substitute for §4.

## 6. Provenance and separation from evaluation

| Safeguard | Mechanism |
|---|---|
| Per-record provenance | every output record carries `provenance.kind = "distilled"`, the provider, the teacher model id and version, the creation date and the request parameters (temperature, prompt template), via `distilled_provenance` (M9) |
| Review status | distilled records enter as `reviewed: false` |
| Run manifest | `distill_manifest.json`: provider, model, start/end time, `terms_reviewed_on`, `acknowledged_by`, temperature, counts requested / kept / rejected (langid, overlap), number of redactions, usage note |
| Language filter | `require_langid_match: true` drops outputs not in the target language/script |
| Eval overlap | `exclude_overlap_with` (default `data/eval/ground_truth.jsonl`) drops outputs that overlap eval prompts; the training contamination guard checks again before training |
| Never in eval data | `data.distill.assert_not_distilled` refuses any file with distilled records as eval ground truth or judge labels (`DistilledDataRejected`); detector gold labels (`responses_gold.jsonl`) are human only |

## 7. Licensing

- The slm-forge **code** is MIT. Distilled **data** is not covered by that licence: it stays subject to the provider's terms. Every manifest carries this usage note:

  > Generated by a third-party teacher model: subject to that provider's terms of use, not covered by the slm-forge MIT licence; do not redistribute by default.

- Do not commit distilled data to a public repository or publish it with a dataset or model unless the provider's terms allow it. Keep it out of `data/eval/`.
- When you publish a model trained on distilled data, state that in the model card, together with the teacher model id and the date of your terms review.

## 8. Checklist

1. Can the same goal be reached with human-written data? If yes, skip distillation.
2. Read the provider's current terms (§3); record the date and your name in `terms_reviewed_on` and `acknowledged_by`.
3. Confirm that seed prompts contain no personal data, or complete the Law 09-08 / CNDP assessment (§4).
4. Keep `redact_pii: true`; set `exclude_overlap_with` to your eval ground truth.
5. Run, then spot-check outputs; treat them as unreviewed training data.
6. Do not redistribute the output by default; mention it in any model card.
