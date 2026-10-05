# hf-preflight

Check a Hugging Face model **before** you download it.

```console
$ hf-preflight org/some-model
RISKY
  org/some-model

    gated: no
    licence: apache-2.0
    size on disk: 13.2 GB across 14 file(s)
  ! remote code: YES — config.json sets auto_map, so loading this model executes
    modeling_custom.py, configuration_custom.py from the repo on your machine
    (requires trust_remote_code=True)
  ! weights: pickle only (pytorch_model.bin) — no safetensors; unpickling
    executes whatever the file says to
```

No dependencies. Python 3.9+. Does not require `huggingface_hub`.

## Why

Four things about a model are worth knowing before the download starts, and all
four are currently learned the hard way:

- **Is it gated?** You find out when the download fails — sometimes after tens
  of gigabytes have already moved.
- **What licence?** `llama3.1` and `other` are not `apache-2.0`. Worth knowing
  before the model is embedded in something you ship.
- **How big is it really?** The model card rarely says. The file list does.
- **Does loading it execute someone else's code on your machine?** Two ways it
  can, both routinely accepted without a thought.

That last one is the reason this exists.

## The two code-execution paths

**`trust_remote_code`.** When a model's `config.json` carries `auto_map`,
`transformers` imports and runs `.py` files *from the model repository* on your
machine. The flag that permits this is passed constantly, usually because a
tutorial said to. `hf-preflight` names the exact files that would run.

**Pickle weights.** `.bin`, `.pt`, `.pth` and `.ckpt` are pickles, and
unpickling executes whatever the pickle instructs. `.safetensors` was created
precisely so this is not true. A repository that ships only pickles is doing
something worth noticing.

Neither is presented as an accusation — plenty of legitimate models use both.
They are reported so the choice to accept them is a choice.

## Install

```bash
pip install hf-preflight
# or, without installing:
uvx hf-preflight org/model
```

## Use

```bash
hf-preflight org/model
hf-preflight https://huggingface.co/org/model     # a pasted URL works
hf-preflight org/model --revision refs/pr/3
hf-preflight org/model --json
```

### Datasets and spaces

The Hub serves models, datasets and spaces from three sibling endpoints that
answer the same shape, so one inspector covers all three:

```bash
hf-preflight stanfordnlp/imdb --type dataset
hf-preflight https://huggingface.co/datasets/stanfordnlp/imdb   # kind read from the URL
hf-preflight https://huggingface.co/spaces/org/demo
```

A pasted URL names its own kind, and that wins over `--type` — a dataset URL
should not need a flag that agrees with it, and must not be quietly inspected
as a model. `--type` defaults to `model`, so every existing invocation is
unchanged, and the kind appears in the output and in `--json` as `repo_type`.

Without this, a real dataset id sent to the models endpoint comes back `401`
and was reported as **"not found"** — the one wrong answer worse than no
answer, because the repository does exist.

Exit codes: `0` clean, `1` risky, `2` blocked, `3` error.

For CI, name the gates you actually care about — this exits non-zero **only**
for those, so an unrelated risky finding does not fail your pipeline:

```bash
hf-preflight "$MODEL" --fail-on remote-code,pickle || {
  echo "refusing to pull a model that executes code on load"; exit 1
}
```

Available gates: `gated`, `license`, `remote-code`, `pickle`.

## Authentication

Optional. Without a token a gated repository is indistinguishable from one that
does not exist, so `hf-preflight` would report "not found" for a model that
merely needs approval.

A token is read from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
`HUGGINGFACEHUB_API_TOKEN`, or `~/.cache/huggingface/token` — so if you have run
`hf auth login` there is nothing to do.

### The token is verified, not assumed

For a gated repo the token is checked against the Hub before the report says
anything about it, because **having a token and having a working one are
different facts**. The Hub serves a repo's metadata identically to a valid
bearer, an expired one and no bearer at all — same `200`, same body — so a stale
token in `~/.cache/huggingface/token` looks exactly like a good one to any check
that only asks whether a token is present.

So the report distinguishes them, and `--json` carries the answer as
`token_state`:

| `token_state` | Meaning | Gated repo verdict |
|---|---|---|
| `valid` | the Hub accepted it (`token_name` is who you are) | `risky` — you still need the licence or approval |
| `invalid` | the Hub rejected it: expired, revoked or malformed | `blocked` — run `hf auth login --force` |
| `unverified` | the check itself could not be completed | `risky` — no claim either way |
| `absent` | no token found | `blocked` — nothing will download |
| `unchecked` | a token exists but the repo is not gated, so it changes nothing | n/a |

That check costs one extra request and is only made where the answer changes a
finding: a gated repo, or a refusal about to be explained. An `invalid` token
also changes the *advice* on a refusal — being told to request access is useless
when the credential is the thing that is broken.

Note that a `valid` token does **not** prove you may download the weights, and
the output says so rather than implying access is confirmed.

## What it does not do

- It does not read the remote `.py` files or judge what they contain. It tells
  you they will run. Reviewing them is your call.
- It does not scan weights for malicious payloads. It tells you the format
  permits them.
- A `clean` verdict means nothing checked here was alarming — not that the model
  is safe.

## Licence

MIT
