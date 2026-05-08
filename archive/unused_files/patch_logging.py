import sys
from pathlib import Path

path = Path("/Users/sarmad/Desktop/Quant-project/trend_signal_engine/run_model_selection.py")
content = path.read_text()

# 1. Add --log-level to ArgumentParser
arg_target = '    parser.add_argument(\n        "--discard_suspicious_models",'
arg_replacement = '    parser.add_argument(\n        "--log-level",\n        type=str,\n        default="INFO",\n        choices=["DEBUG", "INFO", "WARNING", "ERROR"],\n        help="Production log level (default: INFO)",\n    )\n    parser.add_argument(\n        "--discard_suspicious_models",'

if arg_target in content:
    content = content.replace(arg_target, arg_replacement)
else:
    print("Could not find ArgumentParser target")
    sys.exit(1)

# 2. Add logging configuration after parse_args
parse_target = "    args = parser.parse_args()"
parse_replacement = """    args = parser.parse_args()

    # Task 1 & 2: Configure production-grade logging
    _logging.basicConfig(
        level=getattr(_logging, args.log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Log level set to %s", args.log_level.upper())"""

if parse_target in content:
    content = content.replace(parse_target, parse_replacement)
else:
    print("Could not find parse_args target")
    sys.exit(1)

# 3. Replace print statements with logger calls
# Memory checkpoints -> DEBUG
content = content.replace('print(f"  [Memory] {event:<30} | RSS: {rss_gb:.2f} GB")', 'logger.debug("  [Memory] %-30s | RSS: %.2f GB", event, rss_gb)')
content = content.replace('print(f"  [Memory] after_window_{win_idx} | RSS: {PreparedPanelCache.get_rss_mb() / 1024:.2f} GB")', 'logger.debug("  [Memory] after_window_%d | RSS: %.2f GB", win_idx, PreparedPanelCache.get_rss_mb() / 1024)')

# Feature/Target audits -> DEBUG
content = content.replace('print("\\nFeature Polarity Sign-Flow Audit (Task 4):")', 'logger.debug("\\nFeature Polarity Sign-Flow Audit (Task 4):")')
content = content.replace('print("\\nTarget Alignment Audit (Task 2):")', 'logger.debug("\\nTarget Alignment Audit (Task 2):")')
content = content.replace('print("--- FastSweep Performance Audit ---")', 'logger.debug("--- FastSweep Performance Audit ---")')

# Change some other prints to logger.info
content = content.replace('print(f"Phase {phase_name} start")', 'logger.info("Phase %s start", phase_name)')

path.write_text(content)
print("Successfully patched run_model_selection.py for logging")
