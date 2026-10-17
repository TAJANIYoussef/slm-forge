# Protocol locks

A **protocol** is everything that can change a score: scorer, normalizer, matcher and lexicon
versions (with resource hashes), the prompt template and prompt mode, the decoding settings,
τ and the τ sweep, the unanswerable no-cue policy, the utility weights, the primary metrics,
the statistics settings and seed, the dataset hash and split, the subsample ids, and the judge
and detector-validation report (when used). Two results are comparable only if their protocol
hashes are equal. Results are stored under `eval/<suite>/<protocol8>/`, so different protocols
never end up in the same trend or report.

A **lock file** in this directory freezes a protocol before you look at test-split results
(pre-registration). An eval whose live protocol differs from the lock fails with
`ProtocolDrift`, and the message lists the fields that changed.

## Workflow

```bash
# 1. Inspect the live protocol and its hash
slm-forge protocol show -c configs/eval/factual-reliability.yaml

# 2. Tune only on the dev split (lexicons, tau, weights), then freeze
slm-forge protocol lock -c configs/eval/factual-reliability.yaml --out configs/protocols/fr-v1.lock.json

# 3. Point the eval config at the lock and commit both. In the standalone eval config
#    (configs/eval/factual-reliability.yaml) the key is top level:
#      protocol_lock: configs/protocols/fr-v1.lock.json
#    In a run config (configs/<model>.yaml) it sits under the eval: section:
#      eval:
#        protocol_lock: configs/protocols/fr-v1.lock.json
```

## Conventions

- Name a lock `<suite-short>-v<N>.lock.json` (for example `fr-v1.lock.json` for factual
  reliability, version 1). Never edit a lock by hand: write a new version instead.
- Commit the lock in the same change as the config that references it, before running on
  the test split.
- Editing the ground truth, a lexicon, a prompt template or τ changes the protocol. Lock a
  new version and re-run every checkpoint you want to compare; `slm-forge rescore` re-scores
  cached generations without regenerating.
- Results from an unlocked protocol are fine for development. Published test-split numbers
  should come from a locked protocol.

No lock ships with the repository: the protocol depends on your ground-truth file.
