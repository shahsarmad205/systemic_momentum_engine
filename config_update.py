import yaml

with open("backtest_config.yaml", "r") as f:
    config = yaml.safe_load(f)

if "strategy" in config:
    config["strategy"]["holding_period_days"] = 7
    config["strategy"]["min_holding_period_days"] = 3
    config["strategy"]["rebalance_every_trading_days"] = 2

if "regime" in config:
    config["regime"]["bear_skip_new_entries"] = False
    config["regime"]["signal_confidence_multiplier_bear"] = 0.9  
    config["regime"]["signal_confidence_multiplier_sideways"] = 1.0 
    config["regime"]["bear_signal_quantile"] = 0.90
    config["regime"]["crisis_signal_quantile"] = 0.98
    config["regime"]["crisis_risk_mode"] = "cash"
    config["regime"]["crisis_transition_flatten_all"] = True
    config["regime"]["signal_confidence_multiplier_crisis"] = 2.5
    config["regime"]["crisis_block_all_new_entries"] = True

if "risk" in config:
    config["risk"]["take_profit_pct"] = 0.08
    config["risk"]["stop_loss_pct"] = 0.05
    config["risk"]["short_stop_loss_pct"] = 0.03

if "execution_costs" in config:
    config["execution_costs"]["commission_bps"] = 1.0
    config["execution_costs"]["spread_bps"] = 1.0
    config["execution_costs"]["slippage_bps"] = 1.0
    config["execution_costs"]["unit"] = "bps"

with open("backtest_config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print("Config updated.")
