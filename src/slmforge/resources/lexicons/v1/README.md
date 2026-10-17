# Lexicons v1

One YAML file per `<kind>/<bucket>.yaml`:
- kinds: `abstain`, `premise`, `hedge`, `filler`, `contrast`, `stopwords`, `yes_no`
- buckets: `en`, `fr`, `msa`, `darija_arab`, `darija_latn`

`text.lexicons.LexiconSet.load` validates every file and hashes the whole directory into the
evaluation protocol. Editing any file therefore changes the protocol hash, so results scored
with different lexicons are never compared.

## Regex kinds (abstain, premise, hedge, filler, contrast)

```yaml
version: 1
kind: abstain
bucket: en
entries:
  - id: en.abstain.dont_know          # unique across all files
    category: not_know                # free label, reported in breakdowns
    strength: strong                  # strong | weak (weak: courtesy formulas; never a hedge)
    pattern: '\bi (?:do not|don''t) know\b'   # regex on L1 text
    skeleton: null                    # regex on the L2 consonant skeleton (covers both Darija scripts)
    unless_followed_by: '\s+(?:why|whether|if)\b'   # suppress the hit when this follows it
    hedge_kind: null                  # hedge entries only: epistemic | approximation
    note: optional free text
```

Rules:
- **No pattern may match the empty string**, since it would fire on every response.
- **Surface patterns must be written in L1 form.** L1 is casefolded, with no accents, harakat or hamza seats, `ة` written as `ه`, `ى` as `ي`, ASCII digits and ASCII quotes. The loader rejects any literal that could never match.
- **Skeleton patterns must match at least 4 characters** (minimum match width, so alternations and optional parts count as their shortest branch). Short skeletons collide: `momkin` and `makan` both reduce to `mkn`. Use a surface pattern for short cues.
- **Approximation words** (`environ`, `تقريبا`, `t9riban`) are `hedge_kind: approximation`. They are recorded, but they do not count as a hedge, because "environ 70 %" still commits to a value.
- **Domain disclaimers** ("ce n'est pas un conseil juridique") belong to `filler/` with category `disclaimer`. They neither hedge nor abstain.
- **Policy refusals** ("I can't give legal advice") are abstentions with category `policy_refusal` and are reported separately.

## Word kinds

```yaml
# stopwords/<bucket>.yaml
version: 1
kind: stopwords
bucket: fr
words: [le, la, les, ...]

# yes_no/<bucket>.yaml
version: 1
kind: yes_no
bucket: darija_latn
yes: [ah, iyeh, wah]
no: [la, lla]
first_token_only: [la]     # ambiguous words: polarity only at the start of the answer
```

## Editing checklist

1. Add a scoring fixture for the new cue (`tests/fixtures/scoring_cases.jsonl`). Include a hard negative, a phrase the cue must *not* match.
2. Tune cues only on the `dev` split. Never look at `test`.
3. Expect the protocol hash to change. Re-run `slm-forge rescore` on old runs if you want to compare them.
