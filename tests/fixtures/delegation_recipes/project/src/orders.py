"""Order totals for the synthetic review fixture."""


def total_cents(items: list[dict[str, float]]) -> int:
    """Sum price * qty over the items, in whole cents."""
    total = 0
    for item in items:
        total += int(item["price"] * 100) * int(item["qty"])
    return total


def apply_discount(cents: int, percent: int) -> int:
    return cents - cents * percent // 100
