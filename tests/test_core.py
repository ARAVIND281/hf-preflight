"""Unit tests. The Hub is stubbed — the suite makes no network requests."""

from __future__ import annotations

import json

import pytest

from hf_preflight import core
from hf_preflight.cli import main


# --------------------------------------------------------------------------
# id normalisation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "org/model",
        "  org/model  ",
        "https://huggingface.co/org/model",
        "http://huggingface.co/org/model",
        "huggingface.co/org/model",
        "https://huggingface.co/org/model/",
        "https://huggingface.co/org/model/tree/main",
        "https://huggingface.co/org/model/blob/main/config.json",
        "https://huggingface.co/org/model?library=transformers",
    ],
)
def test_normalise_accepts_what_people_paste(raw):
    assert core.normalise_repo_id(raw) == "org/model"


def test_a_legacy_bare_name_is_left_alone_for_the_hub_to_redirect():
    """bert-base-uncased 307s to google-bert/bert-base-uncased; the Hub knows
    the mapping and we should not guess it."""
    assert core.normalise_repo_id("bert-base-uncased") == "bert-base-uncased"


@pytest.mark.parametrize("raw", ["", "   ", "a/b/c/d"])
def test_normalise_rejects_nonsense(raw):
    with pytest.raises(ValueError):
        core.normalise_repo_id(raw)


# --------------------------------------------------------------------------
# byte formatting
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024**3, "5.0 GB")],
)
def test_human_bytes(count, expected):
    assert core.human_bytes(count) == expected


# --------------------------------------------------------------------------
# stubbing the Hub
# --------------------------------------------------------------------------
def _meta(**over):
    meta = {
        "id": "org/model",
        "gated": False,
        "private": False,
        "disabled": False,
        "cardData": {"license": "apache-2.0"},
        "config": {"model_type": "bert", "architectures": ["BertModel"]},
        "siblings": [
            {"rfilename": "config.json", "size": 570},
            {"rfilename": "model.safetensors", "size": 440_000_000},
        ],
    }
    meta.update(over)
    return meta


def _stub(monkeypatch, meta=None, *, token="tok"):
    monkeypatch.setattr(core, "_get", lambda *a, **k: meta or _meta())
    monkeypatch.setattr(core, "_token", lambda: token)


def test_a_plain_open_model_is_clean(monkeypatch):
    _stub(monkeypatch)
    report = core.inspect_model("org/model")
    assert report.severity == "clean"
    assert report.license == "apache-2.0"
    assert report.weight_format == "safetensors"
    assert report.total_bytes == 440_000_570


def test_the_resolved_id_is_reported_when_the_hub_redirects(monkeypatch):
    _stub(monkeypatch, _meta(id="google-bert/bert-base-uncased"))
    report = core.inspect_model("bert-base-uncased")
    assert report.repo_id == "bert-base-uncased"
    assert report.resolved_id == "google-bert/bert-base-uncased"


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------
def test_a_gated_model_without_a_token_is_blocked(monkeypatch):
    _stub(monkeypatch, _meta(gated="manual"), token=None)
    report = core.inspect_model("org/model")
    assert report.severity == "blocked"
    assert "no HF_TOKEN is set" in report.of_kind("gated").detail


def test_a_gated_model_with_a_token_is_risky_not_confirmed(monkeypatch):
    """Reading a gated repo's metadata does not prove you may download its
    weights. Claiming otherwise would be the one lie that costs a download."""
    _stub(monkeypatch, _meta(gated="manual"), token="tok")
    report = core.inspect_model("org/model")
    assert report.severity == "risky"
    detail = report.of_kind("gated").detail
    assert "can read the repo" in detail and "before the weights will download" in detail


def test_auto_gating_says_accept_the_licence_not_request_access(monkeypatch):
    _stub(monkeypatch, _meta(gated="auto"), token="tok")
    assert "accept the licence" in core.inspect_model("org/model").of_kind("gated").detail


# --------------------------------------------------------------------------
# licence
# --------------------------------------------------------------------------
def test_a_permissive_licence_is_clean(monkeypatch):
    _stub(monkeypatch, _meta(cardData={"license": "mit"}))
    assert core.inspect_model("org/model").of_kind("license").severity == "clean"


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("llama3.1", "700M MAU"),
        ("cc-by-nc-4.0", "non-commercial"),
        ("other", "read the model card"),
    ],
)
def test_restrictive_licences_are_flagged_with_the_reason(monkeypatch, name, fragment):
    _stub(monkeypatch, _meta(cardData={"license": name}))
    finding = core.inspect_model("org/model").of_kind("license")
    assert finding.severity == "risky"
    assert fragment in finding.detail


def test_a_missing_licence_is_flagged_not_assumed_open(monkeypatch):
    """No declared licence means all rights reserved, which is the opposite of
    what an empty field looks like."""
    _stub(monkeypatch, _meta(cardData={}))
    finding = core.inspect_model("org/model").of_kind("license")
    assert finding.severity == "risky"
    assert "all rights reserved" in finding.detail


def test_a_list_valued_licence_is_handled(monkeypatch):
    _stub(monkeypatch, _meta(cardData={"license": ["apache-2.0"]}))
    assert core.inspect_model("org/model").license == "apache-2.0"


# --------------------------------------------------------------------------
# remote code — the reason this tool exists
# --------------------------------------------------------------------------
def test_auto_map_means_loading_executes_repo_code(monkeypatch):
    _stub(monkeypatch, _meta(
        config={
            "model_type": "qwen",
            "auto_map": {
                "AutoConfig": "configuration_qwen.QWenConfig",
                "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel",
            },
        },
    ))
    report = core.inspect_model("org/model")
    finding = report.of_kind("remote_code")
    assert finding.severity == "risky"
    assert "configuration_qwen.py" in finding.detail
    assert "modeling_qwen.py" in finding.detail
    assert "trust_remote_code=True" in finding.detail


def test_no_auto_map_means_no_remote_code(monkeypatch):
    _stub(monkeypatch)
    assert core.inspect_model("org/model").of_kind("remote_code").severity == "clean"


def test_auto_map_values_may_be_lists(monkeypatch):
    """Tokenizer entries are ["slow", "fast"] pairs, sometimes with a null."""
    _stub(monkeypatch, _meta(config={
        "auto_map": {"AutoTokenizer": ["tokenization_x.XTokenizer", None]},
    }))
    detail = core.inspect_model("org/model").of_kind("remote_code").detail
    assert "tokenization_x.py" in detail


# --------------------------------------------------------------------------
# weight format
# --------------------------------------------------------------------------
def test_pickle_only_weights_are_flagged(monkeypatch):
    _stub(monkeypatch, _meta(siblings=[
        {"rfilename": "pytorch_model.bin", "size": 500},
    ]))
    report = core.inspect_model("org/model")
    finding = report.of_kind("pickle")
    assert finding.severity == "risky"
    assert "pytorch_model.bin" in finding.detail
    assert report.weight_format == "pickle"


def test_safetensors_alongside_pickles_is_clean(monkeypatch):
    """Loaders prefer safetensors, so the pickle is never opened."""
    _stub(monkeypatch, _meta(siblings=[
        {"rfilename": "model.safetensors", "size": 1},
        {"rfilename": "pytorch_model.bin", "size": 1},
    ]))
    report = core.inspect_model("org/model")
    assert report.of_kind("pickle").severity == "clean"
    assert report.weight_format == "both"


@pytest.mark.parametrize("name", ["a.pt", "a.pth", "a.ckpt", "a.bin"])
def test_every_pickle_extension_is_recognised(monkeypatch, name):
    _stub(monkeypatch, _meta(siblings=[{"rfilename": name, "size": 1}]))
    assert core.inspect_model("org/model").of_kind("pickle").severity == "risky"


# --------------------------------------------------------------------------
# repository status
# --------------------------------------------------------------------------
def test_a_disabled_repo_is_blocked(monkeypatch):
    _stub(monkeypatch, _meta(disabled=True))
    assert core.inspect_model("org/model").severity == "blocked"


def test_a_private_repo_is_noted(monkeypatch):
    _stub(monkeypatch, _meta(private=True))
    assert core.inspect_model("org/model").of_kind("status").severity == "risky"


def test_missing_sizes_do_not_crash_the_total(monkeypatch):
    """?blobs=true is not always honoured; a missing size must not raise."""
    _stub(monkeypatch, _meta(siblings=[
        {"rfilename": "model.safetensors"},
        {"rfilename": "config.json", "size": 10},
    ]))
    assert core.inspect_model("org/model").total_bytes == 10


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_exits_0_for_a_clean_model(monkeypatch, capsys):
    _stub(monkeypatch)
    assert main(["org/model", "--no-colour"]) == 0
    assert "CLEAN" in capsys.readouterr().out


def test_cli_exits_1_for_risky(monkeypatch, capsys):
    _stub(monkeypatch, _meta(config={"auto_map": {"AutoModel": "modeling_x.X"}}))
    assert main(["org/model", "--no-colour"]) == 1


def test_cli_exits_2_for_blocked(monkeypatch, capsys):
    _stub(monkeypatch, _meta(gated="manual"), token=None)
    assert main(["org/model", "--no-colour"]) == 2


def test_cli_exits_3_on_a_hub_error(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise core.HubError("could not reach huggingface.co: timed out")

    monkeypatch.setattr(core, "_get", boom)
    assert main(["org/model", "--no-colour"]) == 3
    assert "error" in capsys.readouterr().err


def test_fail_on_gates_only_the_named_risk(monkeypatch, capsys):
    """A risky licence must not fail a pipeline that only cares about code
    execution — otherwise --fail-on is no better than the plain exit code."""
    _stub(monkeypatch, _meta(cardData={"license": "llama3.1"}))
    assert main(["org/model", "--fail-on", "remote-code", "--no-colour"]) == 0
    assert main(["org/model", "--fail-on", "license", "--no-colour"]) == 2


def test_fail_on_accepts_several_gates(monkeypatch):
    _stub(monkeypatch, _meta(siblings=[{"rfilename": "m.bin", "size": 1}]))
    assert main(["org/model", "--fail-on", "remote-code,pickle", "--no-colour"]) == 2


def test_fail_on_rejects_an_unknown_gate(monkeypatch, capsys):
    _stub(monkeypatch)
    assert main(["org/model", "--fail-on", "nonsense", "--no-colour"]) == 3
    assert "unknown --fail-on" in capsys.readouterr().err


def test_json_output_is_machine_readable(monkeypatch, capsys):
    _stub(monkeypatch, _meta(gated="auto", cardData={"license": "llama3.1"}))
    main(["org/model", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["severity"] == "risky"
    assert payload["gated"] == "auto"
    assert payload["license"] == "llama3.1"
    assert payload["total_human"].endswith("MB")
    assert {f["kind"] for f in payload["findings"]} >= {"gated", "license", "remote_code"}
