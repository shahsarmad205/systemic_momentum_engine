"""
Execution Engine
==================
Applies slippage and transaction costs to trade entries/exits.
"""


class ExecutionEngine:
    """
    Simulates realistic execution frictions.

    Parameters:
        slippage_bps : basis-point slippage applied to each trade side
        commission   : flat USD fee charged per trade side
    """

    def __init__(self, slippage_bps: float = 5.0, commission: float = 1.0):
        self.slippage_frac = slippage_bps / 10_000.0
        self.commission = commission

    def apply_entry_slippage(self, price: float, signal: str) -> float:
        """Worsen the fill price on entry (buy higher, short lower)."""
        if signal == "Bullish":
            return price * (1 + self.slippage_frac)
        return price * (1 - self.slippage_frac)

    def apply_exit_slippage(self, price: float, signal: str) -> float:
        """Worsen the fill price on exit (sell lower, cover higher)."""
        if signal == "Bullish":
            return price * (1 - self.slippage_frac)
        return price * (1 + self.slippage_frac)

    def apply_entry_slippage_scaled(self, price: float, signal: str, scalar: float = 1.0) -> float:
        """VIX-scaled entry slippage. scalar=1 → normal; scalar=2.5 → stress-regime fill.
        Captures the empirical spread expansion during liquidity stress (VIX > rolling median)."""
        frac = self.slippage_frac * max(scalar, 0.0)
        if signal == "Bullish":
            return price * (1.0 + frac)
        return price * (1.0 - frac)

    def apply_exit_slippage_scaled(self, price: float, signal: str, scalar: float = 1.0) -> float:
        """VIX-scaled exit slippage. Symmetric with apply_entry_slippage_scaled."""
        frac = self.slippage_frac * max(scalar, 0.0)
        if signal == "Bullish":
            return price * (1.0 - frac)
        return price * (1.0 + frac)
