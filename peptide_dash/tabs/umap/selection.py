from .shared import *


def _split_variant_tokens(text: Optional[str]) -> List[str]:
    """Split a pasted list of variants (comma/space/newline separated) into tokens."""
    if not text:
        return []
    toks = re.split(r"[\s,;]+", str(text).strip())
    return [t for t in toks if t]


def _match_variants(filter_text: Optional[str], universe: List[str]) -> List[str]:
    """
    Return variants matching the filter.

    - If filter starts with 're:' => treat the remainder as a regex (search).
    - Otherwise => case-insensitive substring match.
    """
    if not filter_text:
        return []
    ft = str(filter_text).strip()
    if not ft:
        return []
    if ft.lower().startswith("re:"):
        pat = ft[3:]
        try:
            rx = re.compile(pat)
        except re.error:
            return []
        return [v for v in universe if rx.search(str(v))]
    needle = ft.lower()
    return [v for v in universe if needle in str(v).lower()]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
