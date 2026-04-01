from backtesting.config import load_backtest_config
cfg = load_backtest_config("backtest_config.yaml")
print("allow_shorts:", cfg.allow_shorts)
print("enable_shorts:", cfg.enable_shorts)
print("long_only:", cfg.long_only)
print("ml:", getattr(cfg, 'ml', None) is not None)
