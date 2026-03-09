# Codebase Review: Proposed Fix Tasks

## 1) Typo fix task
**Task:** Correct README typos and command formatting in the setup/quantization section.

- `modle` should be `model` in the quantization instructions.
- The shell block under virtualenv setup is malformed (`bash` + inline python command formatting).

**Why this matters:** Typos and malformed command blocks reduce trust and can cause copy/paste errors for new contributors.

**References:**
- `README.md` lines 129-132
- `README.md` line 167

---

## 2) Bug fix task
**Task:** Fix text-match UI crash and incorrect match-score rendering in `Pipeline/app.py`.

- `display_product_info` currently requires `match_score`, but text flow calls it with only `product`.
- Match formatting uses `round((match_score)*100),2` which creates a tuple and renders incorrectly.

**Proposed fix direction:**
- Make `match_score` optional (e.g., `match_score: float | None = None`) and render match only when present.
- Correct formatting to `round(match_score * 100, 2)`.

**Why this matters:** Text matching path can throw a `TypeError`, breaking a core user flow.

**References:**
- `Pipeline/app.py` line 12
- `Pipeline/app.py` line 41
- `Pipeline/app.py` line 92

---

## 3) Documentation/comment discrepancy task
**Task:** Align logger module comments/docstring with actual implementation details.

- Module docstring says logs go to a "mocked MongoDB instance," but code connects to `mongodb://localhost:27017`.
- The same section says "separate port/instance," but URI uses default local instance.

**Proposed fix direction:**
- Update wording to reflect real behavior (local MongoDB URI configurable via env var), or actually implement a separate logging URI/instance configuration.

**Why this matters:** Misleading comments create operational confusion during deployment and debugging.

**References:**
- `Pipeline/utils/logger.py` lines 4-6
- `Pipeline/utils/logger.py` line 16

---

## 4) Test improvement task
**Task:** Add tests for Streamlit display helper and matching flow edge cases.

**Recommended test additions:**
- Unit test for `display_product_info` when `match_score` is `None` (text path).
- Unit test for score formatting precision (ensures percentage format is correct).
- Unit test that text flow does not crash when only `input_text` is provided.

**Suggested approach:**
- Refactor `Pipeline/app.py` to move UI-independent logic into small functions that can be tested with `pytest` + mocks.
- Mock `streamlit` calls and matching/inference functions.

**Why this matters:** There are no automated tests currently covering this high-risk UI logic; regressions are likely.

**References:**
- `Pipeline/app.py` lines 12-43
- `Pipeline/app.py` lines 80-93
