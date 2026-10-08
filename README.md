# policy-gap-checker

A small command-line tool that reads a security policy (PDF or TXT), splits it into paragraphs, and asks an LLM to map each paragraph to exactly one of eight hand-picked security controls — or NONE — using validated JSON output. It then prints which controls the policy covers and which ones are missing, and a second mode scores the mapper against a small labeled test set.

> **Note:** labeled_hard.json was drafted by an AI assistant and must be reviewed and relabeled by the author before results are reported.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

python mapper.py --list-models   # print available model ids for your key
export GROQ_MODEL=<model id>     # optional override; defaults to openai/gpt-oss-120b

python mapper.py sample_policy.txt            # coverage report
python mapper.py --eval                       # accuracy on labeled.json (default path)
python mapper.py --eval labeled_hard.json     # accuracy on a custom set
pytest                                        # offline tests, no key needed
```

## How it works

`load_paragraphs()` reads the file (pypdf for `.pdf`, plain UTF-8 otherwise), splits on blank lines, and keeps only chunks longer than 60 characters. Each surviving paragraph is sent to `GROQ_MODEL` (default `openai/gpt-oss-120b`) through the Groq OpenAI-compatible endpoint at temperature 0 with `response_format={"type": "json_object"}`. The reply is parsed and validated by the pydantic `Mapping` model, which rejects any control outside the eight keys plus NONE. Soft failures (bad JSON, validation errors, rate limits) are retried twice and then reported as ERROR, while fatal API errors (404, 401, 403, 400) stop the run with a clear message. The report aggregates real predictions into an OK/MISSING coverage section and reports errors separately.

## Known limitations

- Only one control can be assigned per paragraph, so paragraphs that span several topics are forced into a single label.
- The eight controls are hand-picked rather than drawn from a real framework such as ISO 27001 or NIST CSF.
- The evaluation sets in `labeled.json` (18 items) and `labeled_hard.json` (20 items) are small, so any accuracy figure is noisy.
- LLM output can vary between runs and models; even with temperature 0, labels may shift.
- Paragraph splitting is heuristic (blank lines, with a line-merge fallback for PDFs) and may produce odd chunks.
- Results depend on the chosen model; failed API calls are reported as errors and excluded from accuracy.

## Results

### Starter set (18 clear-cut items, written during development)

Measured on 2026-10-16 with the default model `openai/gpt-oss-120b`:

```
Accuracy: 18/18 = 100% (excluding 0 errors)
Errors: 0 of 18
```

Items are deliberately unambiguous, so this shows the pipeline works, not that the mapper is accurate in general.

### Harder set

Accuracy: TO_BE_FILLED after running: python mapper.py --eval labeled_hard.json (N items, reviewed and labeled by the author)

## Failure cases

TO_BE_WRITTEN by the author: expected X, got Y, because ...

TO_BE_WRITTEN by the author: expected X, got Y, because ...

TO_BE_WRITTEN by the author: expected X, got Y, because ...
