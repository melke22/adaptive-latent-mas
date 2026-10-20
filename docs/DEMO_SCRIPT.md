
# Professor Demo Script

1. Open **Research Dashboard**.
   - State the problem in one sentence.
   - Show Adaptive 62.50%, Fixed-20 56.25%, and the 18.26% step reduction.
   - Emphasize paired significance against Fixed-20.

2. Open **Benchmark Explorer**.
   - Select a case where Fixed-10 fails, Fixed-20 succeeds, and Adaptive succeeds with fewer steps.
   - Explain Planner/Critic/Solver decisions and predicted marginal utility.

3. Open **Controller Analysis**.
   - Show that Planner, Critic, and Solver receive different continuation rates.
   - Explain that the same learned controller is applied role-conditionally.

4. Open **Statistical Evidence**.
   - Explain Wilson intervals, paired bootstrap difference, and exact McNemar test.

5. Open **Architecture**.
   - Explain latent realignment, KV working-memory transfer, zero intermediate text,
     and direct final Solver decoding.

6. Open **Reproducibility**.
   - Show frozen threshold, no test retraining, sealed remainder, continuation verification,
     and 72/72 tests.

7. If asked about future work:
   - State that progressive 5→10→15→20 learned halting is pre-declared and not yet claimed complete.
   - Heterogeneous cross-model alignment is a later extension.
