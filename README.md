# Data science projects

An end-to-end project on real public data. It is self-contained — its own
environment, pipeline, tests and README — and runs from a clean checkout in a
couple of minutes:

```bash
cd projects/nuance && make setup && make all
```

> **Moved:** `projects/overbook` — hotel cancellation risk turned into a
> revenue-optimal overbooking limit — now lives in its own repository:
> [krish-mirpuri/Hotel_Booking_Forecasting](https://github.com/krish-mirpuri/Hotel_Booking_Forecasting).

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

## Conventions

The project is built around one conviction: **the model is the easy part, and
the number you report about it is where the work is.** So it:

- splits on time or on a real distribution shift, never at random;
- fits every decision rule (thresholds, calibrators, category vocabularies) on
  data that is entitled to see it, and proves it with tests;
- attaches uncertainty to the headline, and compares systems in a way that has
  the power to resolve a difference;
- sweeps the assumptions it cannot measure rather than asserting them;
- generates its results document from the artifacts on disk, so no number in a
  README can drift from what the code produced;
- keeps the result that did not go my way in the headline.

Shared conventions: `make setup` / `make all` / `make check`, a
`conf/config.yaml` holding anything that can move a reported number, a
`reports/` directory of generated CSVs, JSON and figures, a FastAPI service and
a dashboard, Docker, and CI across Python 3.10–3.12.

MIT licensed.
