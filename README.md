# Data science projects

Two end-to-end projects on real public data. Each is self-contained — its own
environment, pipeline, tests and README — and each runs from a clean checkout
in a couple of minutes:

```bash
cd projects/<name> && make setup && make all
```

They are deliberately about different halves of the job.

---

## [`projects/overbook`](projects/overbook) — prediction → decision

**Hotels cancel-proof their revenue by selling rooms they don't have. This works
out how many.**

119,210 real bookings from two Portuguese hotels. Predicts per-booking
cancellation risk, calibrates it, and turns it into a nightly authorisation
limit whose value is measured **in euros** on arrivals the model never saw.

- Against the pooled-rate rule hotels actually use: **+426 € per hotel-night**
  (95% CI +312…+554), closing **82%** of the gap to perfect foresight versus
  60% — with a quarter of the walked guests.
- Reported straight: against the *same model* with the decision collapsed to a
  mean, it is **−32 €/night**. Most of the gain is the calibrated
  probabilities, not the optimiser on top.
- Two defects found during evaluation and fixed with past-only information: the
  calibrator drifts as the cancellation rate climbs 33.5% → 41.2%, and arrivals
  are twice as dispersed as independence implies.

*Exact Poisson-binomial arithmetic · asymmetric-loss optimisation · rolling
recalibration · leakage blocklist enforced at runtime · 92 tests.*

---

## [`projects/nuance`](projects/nuance) — measurement → doubt

**27 emotions, 43,410 Reddit comments, and a scoreboard that admits what it
doesn't know.**

Multi-label emotion classification on GoEmotions — and then the harder
question: which of the numbers everyone reports on this benchmark are
measurements, and which are coin flips wearing four decimal places.

- **macro-F1 0.4721, 95% CI [0.4477, 0.4893]** — an interval wider than the gap
  between the methods people rank on this benchmark. `grief` scores 0.500 on
  **six** held-out examples, CI [0.000, 0.800].
- Paired bootstrap resolves a +0.022 difference that separate intervals call a
  wash — and shows that per-label threshold tuning, the standard recipe, is
  **not distinguishable from leaving everything at 0.5** (and the usual version
  of it lands on the wrong side).
- Out of domain, a Reddit-trained sentiment model drops from 0.868 to 0.666 on
  tweets — more than every modelling decision combined. Skipping one
  preprocessing step would have reported 0.794 instead, because the corpus's
  labels are emoticons still sitting in the text.

*Bootstrap-everything via a counts trick · confident-learning label audit from
scratch · domain transfer against VADER · annotation-budget curves · 80 tests.*

---

## What they have in common

Both are built around the same conviction: **the model is the easy part, and
the number you report about it is where the work is.** So both:

- split on time or on a real distribution shift, never at random;
- fit every decision rule (thresholds, calibrators, category vocabularies) on
  data that is entitled to see it, and prove it with tests;
- attach uncertainty to the headline, and compare systems in a way that has
  the power to resolve a difference;
- sweep the assumptions they cannot measure rather than asserting them;
- generate their results document from the artifacts on disk, so no number in
  a README can drift from what the code produced;
- keep the result that did not go my way in the headline.

Shared conventions: `make setup` / `make all` / `make check` in every project,
a `conf/config.yaml` holding anything that can move a reported number, a
`reports/` directory of generated CSVs, JSON and figures, a FastAPI service and
a dashboard, Docker, and CI running both projects across Python 3.10–3.12.

MIT licensed.
