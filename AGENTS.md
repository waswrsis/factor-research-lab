# Factor Research Lab development notes

This repository is a research platform for factor discovery and iterative training.
It is not a trading execution system. See README.md for actual evidence and limits.

- Preserve frozen code, data, environment, protocol and run identities. Changed algorithms need new runs; do not rewrite old failures or refund consumed work.
- Use impact-based regression: test the changed behavior and affected consumers, reuse unchanged evidence with its original source, and do not repeat the full suite for documentation, packaging or a source-hash change.
- Distinguish engineering checks, exposed historical research, independent confirmation and live forward observations. Successful execution is not evidence of predictive advantage.
- Generated formulas execute through the bounded interpreter. Do not grant generated programs arbitrary Python or shell execution.
- Keep caches, datasets, model artifacts, authority stores, credentials and private run records outside Git history. Preserve required evidence before cleaning generated files.
- Windows helpers currently use the D:/FactorResearch project root and the managed environment described in the scripts. Linux CPU dependencies are separate from the Windows/CUDA lock.
- Some engineering tests require frozen fixtures and environments that are not included in the public repository. Report missing prerequisites explicitly; never fabricate a pass or silently relax acceptance.
- A research deployment may run with statistical promotion gates closed. Describe that state honestly; do not present unconfirmed candidates as effective factors.
