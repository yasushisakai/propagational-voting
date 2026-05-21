Python reference implementation of Propagational Proxy Voting.

[arxiv](https://arxiv.org/html/2504.13641v1)

```bash
pip install -r requirements.txt
```

See `examples/basic.py` and `help(ppv.compute)`.

## Benchmarking

`benchmark.csv` records `(version, cell, seed) → wall, RSS`. `tests/snapshots.manifest.json` lists the input/output bundles (hosted as a [GitHub release](https://github.com/yasushisakai/propagational-voting/releases/tag/snapshots-v1), downloaded on demand).

The gatekeeper test confirms an implementation matches the snapshots within `atol=1e-7, rtol=1e-6`:

```bash
pytest tests/test_gatekeeper.py            # fast cells only
pytest tests/test_gatekeeper.py --runslow  # all 21 cells, up to n=20000
```

### Adding a method

One branch per method; branch name == CSV `version` label. `main` holds the current fastest. The new `ppv.py` must keep the `compute(delegates, intermediates, policies, *, tol=1e-9, max_iter=10_000)` signature — the gatekeeper imports `compute` by that name.

```bash
git checkout -b squaring
# write ppv.py with the new algorithm
# in bootstrap_bench.py set VERSION = "squaring"
python bootstrap_bench.py            # appends rows to benchmark.csv
pytest tests/test_gatekeeper.py      # must pass
git commit -am "squaring impl"
git push -u origin squaring
```

When a method beats the current `main`, merge it in and tag the previous one (`baseline-v1`, `squaring-v1`, ...). To accumulate side-by-side comparisons in `benchmark.csv`, cherry-pick each branch's CSV rows back to `main`.

### Adding bigger cells

Edit `CELLS` in `bootstrap_bench.py`, rerun (cached cells skip), then upload:

```bash
gh release upload snapshots-v1 tests/snapshots/n<N>_seed*.npz
```
