# ParGeMSLR Tests

## Test Profiles

- `make test` and `make smoke-test` build the CPU library, run deterministic real Laplacian cases, and check solver-reported and independently recomputed residuals, exact-solution error, and iteration limits.
- `make standard-test` runs the smoke profile across MPI rank and OpenMP settings, checks baseline and same-configuration rerun consistency, and compares convergence metrics and solution vectors against a one-rank CPU baseline.
- Default smoke and standard tolerances are `1e-10` for residual and exact-solution checks with `MAXITS=100` and `KDIM=40`. Override them only for a diagnosed numerical reason.
- Set `MPIRUN_RANK_FLAG=<flag>` when the MPI launcher does not use `-np`; `standard-test` requires a non-empty rank flag so it can vary MPI ranks.
- Set `PARGEMSLR_TEST_CUDA=1` on smoke tests when CUDA hardware and toolchains are available.

## Standard Test Policy

- Do not relax tolerances to pass a failing case without identifying the numerical reason.
- Treat `nan`, `inf`, missing residuals, missing exact-solution errors, missing iteration counts, and nonzero command exits as failures.
- Compare repeated runs at fixed MPI rank and thread counts with high precision, including full solution vectors.
- Compare rank, OpenMP, and optional CUDA variants against the one-rank CPU baseline with convergence-metric, relative solution, and max-absolute solution tolerances.
- Pin BLAS runtime threads in standard-test so OpenMP and MPI coverage is reproducible.
- Require OpenMP-enabled driver logs to report the requested thread count.
- Keep large benchmark matrices outside the default quick path; add them as named regression or nightly cases.
