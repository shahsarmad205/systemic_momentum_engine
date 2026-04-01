import yaml

with open("backtest_config.yaml", "r") as f:
    config = yaml.safe_load(f)

if "strategy" in config:
    config["strategy"]["allow_shorts"] = True
    config["strategy"]["enable_shorts"] = True
    config["strategy"]["long_only"] = False

with open("backtest_config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print("Config updated for shorts.")
