# bardo

Numerical study of **sparse predictive synchronization** in Kuramoto networks.
Each oscillator replaces direct observation of the global mean field with a
Kalman-filtered estimate computed from a fraction $\varepsilon$ of neighbours.
The project measures how little communication suffices for coherent
collective behaviour, and how that bandwidth scales with system size.

Companion to the poster *"The Choir Without a Conductor: Phase Transitions in
Sparse Predictive Multi-Agent Synchronization"* (Atma Anand, Department of
Physics & Astronomy, University of Rochester; Finger Lakes Science &
Technology Showcase, 2026).

## Headline findings

1. Predictive agents reach baseline Kuramoto coherence at $\sim$5 % of the
   all-to-all communication cost ($N=2000$); the fraction shrinks with
   $N$.
2. Sharp phase transition at critical sampling fraction
   $\varepsilon_c(N=2000) \approx 0.0014$; threshold is independent of
   prior precision $K$.
3. Prior precision $K$ sets only the transient timescale:
   $t_{90}\propto K^{0.56}$ empirically.
4. Finite-size scaling: $\varepsilon_c \propto N^{-0.87}$ over
   $N\in\{500,1000,2000,4000\}$, giving sub-quadratic total bandwidth
   $\varepsilon_c N^2 \propto N^{1.13}$.

## Repository layout

```
src/tct/           core simulation modules
  kuramoto.py        all-to-all Kuramoto baseline
  predictive.py      Kalman-filtered agent (matched-condition gain)
  cost.py            interaction + Landauer erasure bookkeeping
  config.py          dataclasses for physics and predictive parameters
  runner.py          multi-seed orchestration
  device.py          torch MPS / CPU selection

experiments/       experiment drivers (one script per exp)
  exp1_baseline_vs_predictive.py
  exp2_phase_transition.py
  exp3_convergence_vs_K.py
  exp4_pareto_K_eps.py
  exp4b_tconv_vs_K.py
  exp5_finite_size_scaling.py

analysis/          plotting + derived analyses
  plot_style.py      Okabe-Ito palette, poster / slide style sheets
  plots.py           build_exp1..5 figure builders
  cost_vs_K.py       out-of-poster check on cost-vs-K scaling

make_figures.py    rebuilds all poster + slide figures from data/*.npz

data/              experiment output (.npz) and per-run manifests
  exp{1..5}_*.npz             numerical results
  exp{1..5}_*_manifest.json   git hash, torch version, timestamp

Anand_Poster_Apr2026.pdf      Poster PDF (beamerposter, 48x36 in)
```

## Reproducing the results

Tested on Apple M4 Pro (MPS backend). Any CUDA or CPU torch install should
also work; the device selector in `src/tct/device.py` falls back to CPU.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the five experiments (each saves to data/*.npz + a manifest)
python experiments/exp1_baseline_vs_predictive.py
python experiments/exp2_phase_transition.py
python experiments/exp3_convergence_vs_K.py
python experiments/exp4_pareto_K_eps.py
python experiments/exp4b_tconv_vs_K.py
python experiments/exp5_finite_size_scaling.py

# rebuild every figure from the .npz files
python make_figures.py

# out-of-poster analysis
python analysis/cost_vs_K.py
```

All runs seed deterministically (three seeds per cell unless noted). Each
run writes a manifest with the git hash, torch version, and timestamp so
results are traceable back to code state.

Default configuration: $N=2000$, $\sigma_\omega=0.5$,
$\sigma_{\text{obs}}=3.0$, $J=1.5$, $dt=0.05$.
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you build on this, please cite the poster:

```
Anand, A. (2026). The Choir Without a Conductor: Phase Transitions in
Sparse Predictive Multi-Agent Synchronization. Finger Lakes Science &
Technology Showcase, University of Rochester.
```
