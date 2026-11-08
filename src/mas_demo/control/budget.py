import math


class BudgetLedger:
    """Single orchestrator owns the ledger; reservations cannot double-spend."""

    def __init__(self, cap, consumed=0):
        if not math.isfinite(cap) or cap <= 0 or not 0 <= consumed:
            raise ValueError("invalid budget")
        self.cap = float(cap)
        self.consumed = float(consumed)
        self.reservations = {}

    @property
    def available(self):
        return max(0.0, self.cap - self.consumed - sum(self.reservations.values()))

    def reserve(self, key, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid reservation")
        if key in self.reservations or seconds > self.available:
            return False
        self.reservations[key] = float(seconds)
        return True

    def settle(self, key, actual_seconds):
        if not math.isfinite(actual_seconds) or actual_seconds < 0:
            raise ValueError("invalid cost")
        self.reservations.pop(key)  # duplicate settlement is an error
        self.consumed += actual_seconds

    def release(self, key):
        self.reservations.pop(key)

    def snapshot(self):
        return {"cap": self.cap, "consumed": self.consumed,
                "reserved": sum(self.reservations.values()), "remaining": self.available,
                "overrun": max(0, self.consumed - self.cap)}
