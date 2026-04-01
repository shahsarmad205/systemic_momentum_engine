import yaml

with open("backtest_config.yaml", "r") as f:
    config = yaml.safe_load(f)

if "execution" in config:
    config["execution"]["allow_shorts"] = True
    config["execution"]["enable_shorts"] = True
    config["execution"]["long_only"] = False
    # Remove Sideways from suppress shorts
    if "regime_suppress_shorts" in config["execution"]:
        config["execution"]["regime_suppress_shorts"] = ["Bull"]
else:
    config["execution"] = {
        "allow_shorts": True,
        "enable_shorts": True,
        "long_only": False,
        "regime_suppress_shorts": ["Bull"]
    }

if "ml" in config and "ensemble" in config["ml"] and "split_models" in config["ml"]["ensemble"]:
    config["ml"]["ensemble"]["split_models"]["enabled"] = True

with open("backtest_config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
