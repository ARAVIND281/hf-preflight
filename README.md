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
`huggingface-cli login` there is nothing to do.

Note that reading a gated repo's metadata does **not** prove you may download
its weights, and the output says so rather than implying access is confirmed.

## What it does not do

- It does not read the remote `.py` files or judge what they contain. It tells
  you they will run. Reviewing them is your call.
- It does not scan weights for malicious payloads. It tells you the format
  permits them.
- A `clean` verdict means nothing checked here was alarming — not that the model
  is safe.

## Licence

MIT
