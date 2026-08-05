# scratch/

One-off scripts: verification snippets, probes, and inspection tools, kept
runnable but outside the human-verified core. This is the **default home for
new AI-generated code** until it earns a place in `src/`.

Nothing in `src/`, `scripts/`, or `tests/` may import from `scratch/`.

Current contents:

- `verify_collection_pipeline.py` — one-shot audit of the collection pipeline contract
- `verify_cup_metric.py` — sanity-check the CuP metric against scripted policies
- `manual_test_constraint_encoder.py` — manual smoke test of the constraint encoder on real weights
