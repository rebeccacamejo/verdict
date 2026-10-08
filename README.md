# verdict

Typed deterministic decision steps for agent toolkits — choice, score, and
yes/no questions answered by Jev-style decision models, with confidence
routing. Declare decisions in YAML or code; the trusted runtime enforces
thresholds, validates outputs, and audits everything.

```bash
pip install git+https://github.com/<you>/verdict.git   # not on PyPI: the
verdict backends                                        # PyPI name belongs
                                                        # to an unrelated project
```

## 60 seconds

```bash
verdict quickstart                            # scaffold + start a real model + sample decision
```

Or step by step:

```bash
verdict init                                   # scaffold verdict.yaml + sample labels
verdict up von                                 # install + start the model server + wire verdict.yaml
verdict doctor                                 # what's actually reachable right now
verdict decide -f verdict.yaml --decision dispute_reason_code \
  --state "I was charged twice for the same order."
verdict eval -f verdict.yaml --dataset eval/labels.jsonl --baseline
```

Every command prints a `→ next:` hint telling you what to run after it.

## Servers: the least-steps story

verdict is the client side — each model runs as its own server. The setup
dance (install → launch → point verdict at it) is automated:

```bash
verdict up <backend>      # install if needed, launch, wait for the port,
                          # pin the URL into verdict.yaml -- one command
verdict ps                # managed servers and their status
verdict down <backend>    # stop it
verdict doctor            # probe every backend: reachable or not, and why
```

After `verdict up von`, plain `verdict decide --backend von ...` just works —
no URL flags: backend URLs resolve automatically (explicit flag >
`<NAME>_BASE_URL` env > `verdict up`-managed server > default).
`verdict eval --dataset labels.jsonl --backends auto` evaluates every
reachable backend head-to-head.

## CLI reference

```bash
verdict init [-o verdict.yaml]                 # scaffold verdict.yaml + eval/labels.jsonl
verdict validate -f verdict.yaml               # validate declarations, run nothing

verdict decide -f verdict.yaml --decision <name> --state "..." [--json]
verdict decide --backend kev --kind choice --instructions "..." \
  --options '{"10.4": "Duplicate processing", "13.1": "Not received"}' \
  --state "..." [--json]
verdict decide --backend kev --kind score --instructions "..." \
  --levels '["none", "weak", "partial", "good", "conclusive"]' --state "..."
verdict decide --backend jev --kind boolean --instructions "Was it charged twice?" \
  --state "..."

verdict backends                               # all 14 backends, auth, notes

verdict up <backend> [--no-install] [--no-wire]  # install + start server + wire YAML
verdict down <backend>                         # stop a managed server
verdict ps                                     # managed servers
verdict doctor                                 # probe every backend

verdict quickstart [--backend von]              # init + up + sample decision, one command

verdict eval -f verdict.yaml --dataset labels.jsonl [--decision <name>] [--baseline] [--json]
verdict eval --dataset labels.jsonl --backends mock,kev,semif --baseline [--json]
verdict eval --dataset labels.jsonl --backends auto   # every reachable backend
```

`--backend-opt base_url=http://host:port/v1/systemone` overrides any backend's
server URL inline; `<NAME>_BASE_URL` env vars do the same (e.g. `KEV_BASE_URL`).

## verdict.yaml

```yaml
version: 1
backends:
  kev:
    base_url: http://localhost:8000/v1/systemone   # per-backend kwargs
decisions:
  - name: dispute_reason_code
    kind: choice                                    # choice | score | boolean
    instructions: "Pick the dispute reason code."
    options:                                        # choice: >= 2 entries
      "10.4": "Duplicate processing"
      "13.1": "Merchandise not received"
    # levels: [...]                                 # score: 2..10 level descriptions
    policy:
      backend: kev                                  # any registered backend
      auto_threshold: 0.85                          # >= auto  -> AUTO_PROCEED
      review_threshold: 0.60                        # >= review -> ESCALATE_TO_LLM
                                                    # below     -> HUMAN_REVIEW
      on_backend_error: human_review                # or escalate_to_llm | fail_closed
```

Why declarative: the output space is fully enumerated up front, the model
generates no text, and the routing policy is static — the declaration *is*
the complete contract. Decisions become versioned, reviewable policy that
the platform owns: validate in CI, evaluate against labeled data, promote.

## Code

```python
from verdict import decide, load_file, Choice, ThresholdPolicy

# One-liner:
r = decide("I was charged twice.",
           Choice(instructions="Pick the reason code.",
                  options={"duplicate": "Charged twice", "other": "Other"}),
           backend="kev")
r.value, r.confidence, r.routing.value   # ("duplicate", 0.93, "auto_proceed")

# From the declarative file:
plan = load_file("verdict.yaml")
r = plan.decide("dispute_reason_code", "I was charged twice.")

# Advanced: typed declarations + explicit runtime (idempotency, audit, replay):
from verdict import DecisionStep, TrustedRuntime, get_backend
step = DecisionStep(name="reason", question=Choice(...),
                    policy=ThresholdPolicy(auto_threshold=0.85, review_threshold=0.60))
rt = TrustedRuntime(get_backend("kev", base_url="http://localhost:8000/v1/systemone"))
out = rt.execute(step, state, idempotency_key="case-4812")
```

## Backends

Backend choice is a **data-classification policy**, not a code change:
PII-bearing case data stays on self-hosted backends.

| Backend | Hosting | License | Notes |
|---|---|---|---|
| `jev` | TypeSafe hosted API | Proprietary | Reference impl. `TYPESAFE_API_KEY`. The one to beat. |
| `kev` | Self-hosted | Apache-2.0 | 0.8/4/9/27B, exact Jev wire, MLX. Most mature open option. |
| `clef-flash` | Self-hosted / Workers AI | Apache-2.0 | Cloudflare 9B. 4-bit quant runs on a 16GB Mac. |
| `strands` | Self-hosted | Apache-2.0 | AWS 2B. Open training recipe. Laptop-CPU friendly. |
| `opendecider` | Self-hosted | Apache-2.0 | nano 400M → 30B MoE. vLLM/Ollama/LM Studio/MLX. |
| `von` | Self-hosted | Apache-2.0 | 395M ModernBERT. The CPU-only pick. |
| `wald` | Self-hosted | Apache-2.0 | 4B, robustness release. |
| `openjev` | Self-hosted | MIT | 4B, non-autoregressive head. |
| `mapika` | Self-hosted | Apache-2.0 | 2B/4B/35B-A3B. CUDA graph capture. |
| `laya` | Self-hosted (`laya-serve`) | Apache-2.0 | 421M + multilingual + fine-tuned ckpts. Fine-tune on your domain. |
| `clef` | Self-hosted / Workers AI | Apache-2.0 | Cloudflare 27B, multimodal. Data-center class. |
| `clm` | Self-hosted | Apache-2.0 | Thinnest docs here — eval before trusting. |
| `semif` | Local harness | MIT | Research harness, not a server. For study/evals. |
| `mock` | In-process | — | Zero-config test double. |

Adding a backend: one entry in `verdict/backends/__init__.py`. If it speaks
`POST /v1/systemone`, subclass `SystemOneHttpBackend` (~20 lines, see `kev.py`).
The CLI, `decide()`, `load_file()`, and eval pick it up automatically.

## Architecture

```
declare  ->  SDK / verdict.yaml     frozen declarations; the SDK enforces nothing
execute  ->  TrustedRuntime         1. re-validates declaration
                                    2. idempotency check
                                    3. backend call (timed)
                                    4. schema-validates the raw answer --
                                       malformed outputs raise before routing
                                    5. confidence routing (policy)
                                    6. audit record (replayable)
              Backend adapter        registry: jev / kev / clef-flash / ... / mock
```

Rule of thumb: if the possible answers are known in advance, it belongs in a
decision step. If it needs prose or open-ended reasoning, it stays with the LLM.

## License

Apache 2.0. See `LICENSE`.
