"""policy-gap-checker: map policy paragraphs to 8 security controls with an LLM."""
import json
import os
import re
import sys

import openai
from pypdf import PdfReader
from pydantic import BaseModel, field_validator

CONTROLS = ["ACCESS", "ENCRYPT", "LOGGING", "INCIDENT", "BACKUP", "VENDOR", "CHANGE", "TRAINING"]
VALID = set(CONTROLS) | {"NONE", "ERROR"}  # ERROR is our internal sentinel, never offered to the model
MIN_LEN = 60
BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # best JSON-capable chat model on this key; GROQ_MODEL overrides
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL = os.path.join(HERE, "labeled.json")
# Fatal API errors: no retry, never swallowed.
FATAL = (openai.NotFoundError, openai.AuthenticationError,
         openai.PermissionDeniedError, openai.BadRequestError)
SYSTEM = (
    "Map the policy paragraph to exactly one control. "
    "ACCESS (least privilege, MFA, access reviews); ENCRYPT (encryption at rest and in transit); "
    "LOGGING (logging and monitoring); INCIDENT (incident response and breach notification); "
    "BACKUP (backups and disaster recovery); VENDOR (third-party and vendor risk); "
    "CHANGE (change management and secure development); TRAINING (security awareness training); "
    "NONE (matches no control). "
    'Reply with JSON only: {"control": "<KEY or NONE>", "reason": "<one sentence>"}'
)


class Mapping(BaseModel):
    """One paragraph mapped to exactly one control."""
    control: str
    reason: str

    @field_validator("control")
    @classmethod
    def control_in_allowlist(cls, v: str) -> str:
        # Reject anything outside the 8 controls, NONE and the ERROR sentinel.
        if v not in VALID:
            raise ValueError(f"control must be one of {sorted(VALID)}")
        return v


def _client():
    """Build the OpenAI client lazily so tests never need an API key."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return openai.OpenAI(base_url=BASE_URL, api_key=key)


def classify(paragraph: str, retries: int = 2) -> Mapping:
    """Label one paragraph; fatal errors propagate, soft failures become ERROR."""
    model = os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    last = "unknown error"
    for _ in range(retries + 1):
        try:
            resp = _client().chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": paragraph}],
            )
            return Mapping(**json.loads(resp.choices[0].message.content))
        except FATAL:
            raise  # 404/401/403/400 will not change on retry
        except Exception as err:  # bad JSON, validation, rate limit, timeout, connection
            last = str(err)
    return Mapping(control="ERROR", reason=f"failed after retries: {last}")


def _blank_split(text: str) -> list:
    return [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]


def _merge_lines(text: str) -> list:
    """PDF fallback: merge single lines until a chunk clears the length bar."""
    out, buf = [], ""
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        buf = f"{buf} {line}".strip()
        if len(buf) > MIN_LEN:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def load_paragraphs(path: str) -> list:
    """Read a .txt/.pdf policy and keep only chunks longer than 60 characters."""
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as fh:
            text = "\n".join((p.extract_text() or "") for p in PdfReader(fh).pages)
        chunks = _blank_split(text)
        if len(chunks) < 2:  # many PDFs have no blank lines between paragraphs
            chunks = _merge_lines(text)
    else:
        with open(path, encoding="utf-8") as fh:
            chunks = _blank_split(fh.read())
    return [c for c in chunks if len(c) > MIN_LEN]


def run_report(path: str) -> None:
    """Print one line per paragraph, then which controls are covered."""
    paras = load_paragraphs(path)
    if not paras:
        print("No paragraphs longer than 60 characters found.")
        return
    counts = dict.fromkeys(CONTROLS, 0)
    errors = 0
    for i, para in enumerate(paras, 1):
        m = classify(para)
        print(f"[{i}/{len(paras)}] {m.control} {m.reason}")
        if m.control == "ERROR":
            errors += 1
        elif m.control in counts:  # ERROR paragraphs never count as coverage
            counts[m.control] += 1
    print("\nCOVERAGE")
    for ctrl in CONTROLS:
        flag = "OK" if counts[ctrl] else "MISSING"
        print(f"{ctrl:9} {flag:7} {counts[ctrl]}")
    print(f"Errors: {errors} of {len(paras)} paragraphs")
    if errors == len(paras):
        print("No paragraphs were classified successfully.")
        sys.exit(1)


def run_eval(path: str = DEFAULT_EVAL) -> None:
    """Score classify() against a labeled JSON file, excluding errors."""
    if not os.path.isfile(path):
        print(f"Error: eval file not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"Evaluating: {os.path.basename(path)}")
    right, errors, wrong = 0, 0, []
    for item in data:
        m = classify(item["text"])  # extra fields such as "note" are ignored
        if m.control == "ERROR":
            errors += 1
        elif m.control == item["control"]:
            right += 1
        else:
            wrong.append((item["control"], m.control, item["text"]))
    scored = len(data) - errors
    if scored == 0:
        print("No items were classified successfully.")
        sys.exit(1)
    print(f"Accuracy: {right}/{scored} = {100.0 * right / scored:.0f}% (excluding {errors} errors)")
    print(f"Errors: {errors} of {len(data)}")
    for exp, got, text in wrong:
        print(f"expected {exp} got {got}: {text[:80]}")
    if errors:
        print("WARNING: results incomplete, do not report this accuracy.")


def parse_args(args: list):
    """Split argv into (mode, path); mode is report/eval/list-models, or None."""
    if args == ["--list-models"]:
        return "list-models", None
    if args and args[0] == "--eval":
        if len(args) > 2:  # --eval takes at most one path
            return None, None
        return "eval", args[1] if len(args) == 2 else DEFAULT_EVAL
    if len(args) == 1:
        return "report", args[0]
    return None, None


def main() -> None:
    mode, path = parse_args(sys.argv[1:])
    if mode is None:
        print("usage: python mapper.py <policy file>  |  --eval [labeled.json]  |  --list-models")
        sys.exit(2)
    # Only the real run path needs a key; tests never call main().
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set. Run: export GROQ_API_KEY=<your key>")
        sys.exit(1)
    if mode == "list-models":
        for m in _client().models.list().data:  # needs a key, not a model
            print(m.id)
        return
    try:
        run_eval(path) if mode == "eval" else run_report(path)
    except FATAL as err:
        print(f"Fatal API error: {err}. Check GROQ_API_KEY and GROQ_MODEL (run --list-models).")
        sys.exit(1)


if __name__ == "__main__":
    main()
