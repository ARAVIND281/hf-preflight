"""Inspect a Hugging Face model repository before downloading it.

Four things about a model are worth knowing *before* the download starts, and
all four are currently learned the hard way:

1. **Is it gated?** Llama and friends need an approval request. You find out
   after the download fails, or after 40 GB have already moved.
2. **What licence?** ``llama3.1`` and ``other`` are not ``apache-2.0``. The
   difference matters before the model is embedded in something shipped.
3. **How big is it, actually?** The card rarely says. The file list does.
4. **Does loading it run somebody else's code on your machine?** Two ways it
   can, both routinely accepted without a thought:

   * ``trust_remote_code`` — a ``config.json`` carrying ``auto_map`` points at
     ``.py`` files in the repository, which ``transformers`` imports and
     executes locally.
   * **pickle weights** — ``.bin`` / ``.pt`` / ``.pth`` are pickles, and
     unpickling executes whatever the pickle says to. ``.safetensors`` exists
     precisely so this is not true.

A fifth thing turns out to matter as much as any of them: **does the token you
have actually work?** The Hub answers a public metadata request identically for
a valid bearer, an expired one and no bearer at all — same 200, same body — so
"a token was found on disk" is not evidence of anything. For a gated repo the
token is therefore verified against ``whoami-v2`` before the report claims it
can read anything; see :class:`TokenStatus`.

Only the public Hub API is used, through :mod:`urllib`, so there are no runtime
dependencies and no dependency on ``huggingface_hub`` being installed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

__all__ = [
    "Finding",
    "Report",
    "TokenStatus",
    "inspect_model",
    "HubError",
    "SEVERITIES",
    "TOKEN_STATES",
]

_API = "https://huggingface.co/api/models"
# The cheapest question that distinguishes "I have a token" from "my token works".
_WHOAMI = "https://huggingface.co/api/whoami-v2"

# Weight formats that execute code when loaded. safetensors was introduced to
# avoid exactly this, so a repo offering only pickles is a deliberate signal.
_PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt")
_SAFE_SUFFIXES = (".safetensors",)

# Licences that are not what people assume "open" means. Not a judgement — a
# prompt to read before shipping something built on them.
_RESTRICTIVE = {
    "llama2": "Meta Llama 2 licence — commercial limits above 700M MAU",
    "llama3": "Meta Llama 3 licence — commercial limits above 700M MAU",
    "llama3.1": "Meta Llama 3.1 licence — commercial limits above 700M MAU",
    "llama3.2": "Meta Llama 3.2 licence — commercial limits above 700M MAU",
    "llama3.3": "Meta Llama 3.3 licence — commercial limits above 700M MAU",
    "cc-by-nc-4.0": "non-commercial only",
    "cc-by-nc-sa-4.0": "non-commercial, share-alike",
    "cc-by-nc-nd-4.0": "non-commercial, no derivatives",
    "gemma": "Gemma terms — use restrictions apply",
    "other": "non-standard licence — read the model card",
    "unknown": "no licence declared — default is all rights reserved",
}

SEVERITIES = ("clean", "risky", "blocked")

# What we know about the token we found. "unchecked" is not ignorance we are
# hiding: for an ungated repo the token's validity changes no finding, and
# spending a request to establish it anyway would be a cost with no answer
# attached.
TOKEN_ABSENT = "absent"
TOKEN_UNCHECKED = "unchecked"
TOKEN_INVALID = "invalid"
TOKEN_UNVERIFIED = "unverified"
TOKEN_VALID = "valid"
TOKEN_STATES = (
    TOKEN_ABSENT,
    TOKEN_UNCHECKED,
    TOKEN_INVALID,
    TOKEN_UNVERIFIED,
    TOKEN_VALID,
)


class HubError(RuntimeError):
    """The Hub could not be reached, or refused the request."""


@dataclass
class Finding:
    """One thing worth knowing before downloading."""

    kind: str          # gated | license | size | remote_code | pickle | status
    severity: str      # clean | risky | blocked
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass
class TokenStatus:
    """Whether the token we found is one the Hub actually accepts.

    The distinction this type exists to keep is between *having* a token and
    *having a working one*. An expired or revoked token is still a string in
    ``~/.cache/huggingface/token``, and the Hub serves a public repo's metadata
    to an invalid bearer exactly as it does to an anonymous caller — same 200,
    same body. So a report that infers "authenticated" from "a token was found"
    will tell you your token can read a gated repo at the moment the Hub is in
    fact rejecting it, which is the one reading that costs a download.
    """

    state: str
    name: str | None = None

    @property
    def usable(self) -> bool:
        """True only when the Hub confirmed the token. Unverified is not a yes."""
        return self.state == TOKEN_VALID

    @property
    def phrase(self) -> str:
        """How to refer to the token in a sentence, naming the user if known."""
        return f"your token ({self.name})" if self.name else "your token"


@dataclass
class Report:
    """Everything learned about a model repository."""

    repo_id: str
    resolved_id: str
    findings: list[Finding] = field(default_factory=list)
    total_bytes: int = 0
    weight_format: str = "unknown"
    license: str | None = None
    gated: str | None = None
    token: TokenStatus = field(default_factory=lambda: TokenStatus(TOKEN_UNCHECKED))

    @property
    def severity(self) -> str:
        """The worst thing found."""
        for level in reversed(SEVERITIES):
            if any(f.severity == level for f in self.findings):
                return level
        return "clean"

    def of_kind(self, kind: str) -> Finding | None:
        for finding in self.findings:
            if finding.kind == kind:
                return finding
        return None


def _token() -> str | None:
    """A Hub token from the environment or the standard cache location.

    Without one, a gated repository looks identical to a missing one, so the
    tool would report "not found" for a model that merely needs approval.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    for path in (
        os.environ.get("HF_TOKEN_PATH"),
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
    ):
        if not path:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                token = handle.read().strip()
            if token:
                return token
        except OSError:
            continue
    return None


def _check_token(token: str | None) -> TokenStatus:
    """Ask the Hub whether ``token`` is accepted, and as whom.

    One extra request, and only made where the answer changes what we say: a
    gated repo, or a refusal we are about to explain. ``whoami-v2`` is the right
    endpoint because it is the only one that *must* authenticate — a repo
    endpoint answers 200 for a public repo whatever the bearer says, so it
    cannot be used to test a token.
    """
    if not token:
        return TokenStatus(TOKEN_ABSENT)
    request = urllib.request.Request(_WHOAMI)
    request.add_header("User-Agent", "hf-preflight")
    request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 401/403 from whoami is the Hub saying the credential itself is no
        # good. Anything else is the Hub having a bad day, which is not a
        # verdict on the token.
        if exc.code in (401, 403):
            return TokenStatus(TOKEN_INVALID)
        return TokenStatus(TOKEN_UNVERIFIED)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        # Could not ask. Reporting "invalid" on a flaky connection would send
        # someone off to replace a token that was fine.
        return TokenStatus(TOKEN_UNVERIFIED)
    if not isinstance(payload, dict):
        return TokenStatus(TOKEN_UNVERIFIED)
    name = payload.get("name") or payload.get("fullname")
    return TokenStatus(TOKEN_VALID, name=str(name) if name else None)


def _access_error(repo_hint: str, token: str | None) -> str:
    """Explain a 401/403/404, resolving which of three things is actually wrong.

    The Hub deliberately does not distinguish "does not exist" from "you may not
    see it", so all three codes arrive here looking identical. What we *can*
    settle is whether the token is the problem, and the three cases need
    different actions from the reader: get a token, replace a broken one, or go
    request access. Telling someone whose token has expired to request access is
    advice that cannot work.
    """
    if token is None:
        return (
            f"{repo_hint} not found — check the id, or set HF_TOKEN if it is "
            "private or gated"
        )
    status = _check_token(token)
    if status.state == TOKEN_INVALID:
        return (
            f"the token you have is not valid — the Hub rejected it, so "
            f"{repo_hint} cannot be read whether or not you have access to it. "
            "Run `hf auth login --force` to replace it, then try again"
        )
    if status.state == TOKEN_UNVERIFIED:
        return (
            f"{repo_hint} not found, or your token does not have access to it "
            "(the token could not be verified — the Hub did not answer the check)"
        )
    return (
        f"{repo_hint} not found, or {status.phrase} does not have access to it "
        "(gated repos need access granted, not just a valid token)"
    )


def _get(path: str, *, repo_hint: str = "that model") -> dict:
    """GET a Hub API path, following the redirect legacy ids produce.

    ``bert-base-uncased`` now 307s to ``google-bert/bert-base-uncased``; without
    following it the response body is the redirect notice, not JSON.
    """
    request = urllib.request.Request(f"{_API}/{path}")
    request.add_header("User-Agent", "hf-preflight")
    token = _token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 401/403/404 all mean "no data for you" and the Hub will not say which.
        # _access_error spends one more request to work out whether the token is
        # the reason, because the remedy differs completely.
        if exc.code in (401, 403, 404):
            raise HubError(_access_error(repo_hint, token)) from exc
        raise HubError(f"the Hub returned {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise HubError(f"could not reach huggingface.co: {exc.reason}") from exc


def human_bytes(count: int) -> str:
    """Bytes as something a person can judge a download by."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _gated_finding(meta: dict, token: TokenStatus) -> Finding:
    """Gated repos are the most common wasted download.

    The metadata coming back proves nothing about the token: the Hub serves a
    gated repo's *card* to anyone. So what is said here turns on whether the Hub
    confirmed the credential, not on whether one was found on disk.
    """
    gated = meta.get("gated")
    if not gated:
        return Finding("gated", "clean", "gated: no")
    how = "accept the licence" if gated == "auto" else "request access and be approved"
    if token.state == TOKEN_VALID:
        # A confirmed token still does not prove you may download weights, so
        # this stays "risky" rather than claiming access.
        return Finding(
            "gated",
            "risky",
            f"gated: {gated} — {token.phrase} is valid and can read the repo, "
            f"but you must {how} before the weights will download",
        )
    if token.state == TOKEN_INVALID:
        # Nothing will download, and the reason is not the gate.
        return Finding(
            "gated",
            "blocked",
            f"gated: {gated} — you must {how}, and the token you have is not "
            "valid: the Hub rejected it, so run `hf auth login --force`",
        )
    if token.state == TOKEN_UNVERIFIED:
        return Finding(
            "gated",
            "risky",
            f"gated: {gated} — you must {how} before the weights will download; "
            "your token could not be verified (the Hub did not answer the check)",
        )
    return Finding(
        "gated",
        "blocked",
        f"gated: {gated} — you must {how}, and no HF_TOKEN is set",
    )


def _license_finding(meta: dict) -> Finding:
    card = meta.get("cardData") or {}
    name = card.get("license")
    if isinstance(name, list):
        name = name[0] if name else None
    if not name:
        note = _RESTRICTIVE["unknown"]
        return Finding("license", "risky", f"licence: none declared — {note}")
    note = _RESTRICTIVE.get(str(name).lower())
    if note:
        return Finding("license", "risky", f"licence: {name} — {note}")
    return Finding("license", "clean", f"licence: {name}")


def _remote_code_finding(meta: dict, py_files: list[str]) -> Finding:
    """``auto_map`` means transformers will import code from the repo."""
    config = meta.get("config") or {}
    auto_map = config.get("auto_map")
    if not auto_map:
        return Finding("remote_code", "clean", "remote code: no")
    targets = sorted({str(v).split(".")[0] + ".py" for v in _flatten(auto_map)})
    named = ", ".join(targets[:4]) or "files in the repo"
    return Finding(
        "remote_code",
        "risky",
        f"remote code: YES — config.json sets auto_map, so loading this model "
        f"executes {named} from the repo on your machine "
        f"(requires trust_remote_code=True)",
    )


def _flatten(auto_map: object) -> list[str]:
    """auto_map values are strings, or lists of them."""
    out: list[str] = []
    if isinstance(auto_map, dict):
        for value in auto_map.values():
            out.extend(_flatten(value))
    elif isinstance(auto_map, (list, tuple)):
        for value in auto_map:
            out.extend(_flatten(value))
    elif isinstance(auto_map, str):
        out.append(auto_map)
    return out


def _weight_findings(files: list[str]) -> tuple[Finding, str]:
    """Pickle weights execute code on load; safetensors do not."""
    pickles = [f for f in files if f.endswith(_PICKLE_SUFFIXES)]
    safe = [f for f in files if f.endswith(_SAFE_SUFFIXES)]
    if safe and not pickles:
        return Finding("pickle", "clean", "weights: safetensors"), "safetensors"
    if safe and pickles:
        return (
            Finding(
                "pickle",
                "clean",
                f"weights: safetensors available (also {len(pickles)} pickle "
                f"file(s), which loaders will skip)",
            ),
            "both",
        )
    if pickles:
        shown = ", ".join(sorted(pickles)[:3])
        return (
            Finding(
                "pickle",
                "risky",
                f"weights: pickle only ({shown}) — no safetensors; unpickling "
                f"executes whatever the file says to",
            ),
            "pickle",
        )
    return Finding("pickle", "clean", "weights: no weight files found"), "none"


def inspect_model(repo_id: str, *, revision: str | None = None) -> Report:
    """Gather everything worth knowing before downloading ``repo_id``.

    Args:
        repo_id: ``org/name``, or a legacy bare name, or a huggingface.co URL.
        revision: branch, tag or commit. Defaults to the repo's default branch.

    Returns:
        A :class:`Report`. Its ``severity`` is the worst finding: ``clean``,
        ``risky`` (something executes code, or the licence is restrictive), or
        ``blocked`` (you cannot download it at all as configured).
    """
    repo_id = normalise_repo_id(repo_id)
    path = urllib.parse.quote(repo_id, safe="/")
    if revision:
        path = f"{path}/revision/{urllib.parse.quote(revision, safe='')}"
    meta = _get(f"{path}?blobs=true", repo_hint=repo_id)

    report = Report(repo_id=repo_id, resolved_id=meta.get("id") or repo_id)
    report.gated = meta.get("gated") or None
    card = meta.get("cardData") or {}
    lic = card.get("license")
    report.license = lic[0] if isinstance(lic, list) and lic else lic

    if meta.get("disabled"):
        report.findings.append(
            Finding("status", "blocked", "the repository is disabled on the Hub")
        )
    if meta.get("private"):
        report.findings.append(Finding("status", "risky", "the repository is private"))

    # Only worth a request when the repo is gated; for an open repo the token's
    # validity changes no finding here.
    report.token = (
        _check_token(_token())
        if report.gated
        else TokenStatus(TOKEN_ABSENT if _token() is None else TOKEN_UNCHECKED)
    )
    report.findings.append(_gated_finding(meta, report.token))
    report.findings.append(_license_finding(meta))

    siblings = meta.get("siblings") or []
    files = [s.get("rfilename", "") for s in siblings]
    report.total_bytes = sum(int(s.get("size") or 0) for s in siblings)
    report.findings.append(
        Finding("size", "clean", f"size on disk: {human_bytes(report.total_bytes)} "
                                 f"across {len(files)} file(s)")
    )

    py_files = [f for f in files if f.endswith(".py")]
    report.findings.append(_remote_code_finding(meta, py_files))

    weight_finding, fmt = _weight_findings(files)
    report.weight_format = fmt
    report.findings.append(weight_finding)
    return report


def normalise_repo_id(raw: str) -> str:
    """Accept ``org/name``, a bare legacy name, or a huggingface.co URL."""
    value = raw.strip()
    if not value:
        raise ValueError("give a model id, e.g. google-bert/bert-base-uncased")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.split("?")[0].split("#")[0].rstrip("/")
    # A URL may carry /tree/main or /blob/main/... after the id.
    parts = value.split("/")
    if len(parts) > 2 and parts[2] in {"tree", "blob", "resolve"}:
        value = "/".join(parts[:2])
    if value.count("/") > 1:
        raise ValueError(f"{raw!r} does not look like a model id")
    return value
