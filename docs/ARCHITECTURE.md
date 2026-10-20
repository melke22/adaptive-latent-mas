
# Local / Kaggle / GitHub Architecture

```text
                         ┌──────────────────────────────┐
                         │          GitHub              │
                         │ code · docs · tests · CI     │
                         └──────────────┬───────────────┘
                                        │ git pull/push
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Local workstation                           │
│                                                                     │
│   Streamlit UI ─── Results Mode ─── tracked compact Phase-9 data    │
│       │                                                             │
│       └──────── Kaggle CLI ────────────────┐                        │
└─────────────────────────────────────────────┼───────────────────────┘
                                              │ auth / push / status /
                                              │ output download
                                              ▼
                         ┌──────────────────────────────┐
                         │           Kaggle             │
                         │ GPU jobs · notebooks         │
                         │ Phase-8 persistent dataset   │
                         │ large research artifacts     │
                         └──────────────────────────────┘
```

The local UI must never rely on Kaggle being available for the already completed Phase-9 demo.
That evidence is stored locally in compact CSV form so the professor demonstration is immediate.

For new heavy inference, Kaggle acts as the compute service. The Kaggle CLI supports kernel
push, status inspection, logs, and output download.
