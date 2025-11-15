# original prompt:

```
Task:
Given globals text (str) and k (int), produce the Top-K most frequent tokens.

Tokenization:
- Case-insensitive tokenization using an ASCII regex; produce lowercase tokens. Whole-string lowercasing is not required.
- Tokens are ASCII [a-z0-9]+ sequences; treat all other characters as separators.

Output:
- Define top_k as a list of (token, count) tuples.
- Sort by count desc, then token asc.
- Length = min(k, number of unique tokens).

Notes:
- Run as-is with the provided globals; no file or network I/O.
```

# optimized prompt:

```md
# Objective
Generate a single, self-contained Python script that exactly solves the specified task on a MacBook Pro (M4 Max).

# Hard requirements
- Use only Python stdlib. No approximate algorithms.
- Tokenization: ASCII [a-z0-9]+ on the original text; match case-insensitively and lowercase tokens individually. Do NOT call text.lower() on the full string.
- Exact Top‑K semantics: sort by count desc, then token asc. No reliance on Counter.most_common tie behavior.
- Define `top_k` as a list of (token, count) tuples with length = min(k, number of unique tokens).
- When globals `text` (str) and `k` (int) exist, do not reassign them; set `top_k` from those globals. If you include a `__main__` demo, guard it to run only when globals are absent.
- No file I/O, stdin, or network access, except optionally printing `top_k` as the last line.

# Performance & memory constraints
- Do NOT materialize the entire token stream or any large intermediate list.
- Do NOT sort all unique (token, count) items unless k >= 0.3 * number_of_unique_tokens.
- When k < number_of_unique_tokens, compute Top‑K using a bounded min‑heap of size k over counts.items(), maintaining the correct tie-break (count desc, then token asc).
- Target peak additional memory beyond the counts dict to O(k). Avoid creating `items = sorted(counts.items(), ...)` for large unique sets.

# Guidance
- Build counts via a generator over re.finditer with re.ASCII | re.IGNORECASE; lowercase each matched token before counting.
- Prefer heapq.nsmallest(k, cnt.items(), key=lambda kv: (-kv[1], kv[0])) for exact selection without full sort; avoid heapq.nlargest.
- Do NOT wrap tokens in custom comparator classes (e.g., reverse-lex __lt__) or rely on tuple tricks for heap ordering.
- Keep comments minimal; include a brief complexity note (time and space).

# Output format
- Output only one Python code block; no text outside the block.

# Examples 
    ```python
    import re, heapq
    from collections import Counter
    from typing import List, Tuple, Iterable

    _TOKEN = re.compile(r"[a-z0-9]+", flags=re.ASCII | re.IGNORECASE)

    def _tokens(s: str) -> Iterable[str]:
        # Case-insensitive match; lowercase per token to avoid copying the whole string
        for m in _TOKEN.finditer(s):
            yield m.group(0).lower()

    def top_k_tokens(text: str, k: int) -> List[Tuple[str, int]]:
        if k <= 0:
            return []
        cnt = Counter(_tokens(text))
        u = len(cnt)
        key = lambda kv: (-kv[1], kv[0])
        if k >= u:
            return sorted(cnt.items(), key=key)
        # Exact selection with bounded memory
        return heapq.nsmallest(k, cnt.items(), key=key)

    # Compute from provided globals when available; demo only if missing and running as main
    try:
        text; k  # type: ignore[name-defined]
    except NameError:
        if __name__ == "__main__":
            demo_text = "A a b b b c1 C1 c1 -- d! d? e"
            demo_k = 3
            top_k = top_k_tokens(demo_text, demo_k)
            print(top_k)
    else:
        top_k = top_k_tokens(text, k)  # type: ignore[name-defined]
    # Complexity: counting O(N tokens), selection O(U log k) via heapq.nsmallest; extra space O(U + k)
    ```
```