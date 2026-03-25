"""Broker integrations (paper / live)."""

__all__ = ["AlpacaBroker"]


def __getattr__(name: str):
    if name == "AlpacaBroker":
        from .alpaca_broker import AlpacaBroker

        return AlpacaBroker
    raise AttributeError(name)
