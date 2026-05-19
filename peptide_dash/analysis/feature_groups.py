from __future__ import annotations

from typing import Dict, List


def is_technical(col: str) -> bool:
    """Heuristic: treat known technical prefixes as 'technical' columns."""
    c = str(col)
    tech_prefixes = ("mbar_", "ESS_", "tau_int_", "jsd2_", "cv_", "psd_", "bootstrap_")
    return c.startswith(tech_prefixes)


def group_columns_by_prefix(cols: List[str]) -> Dict[str, List[str]]:
    """Group columns by the text before the first '__'. Falls back to 'misc'."""
    groups: Dict[str, List[str]] = {}
    for c in cols:
        c = str(c)
        if "__" in c:
            g = c.split("__", 1)[0]
        else:
            g = "misc"
        groups.setdefault(g, []).append(c)
    return groups
