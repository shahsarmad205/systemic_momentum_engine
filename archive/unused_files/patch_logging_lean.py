import sys
from pathlib import Path

path = Path("/Users/sarmad/Desktop/Quant-project/trend_signal_engine/run_model_selection.py")
content = path.read_text()

# Add CLI flag
if 'parser.add_argument("--log-level"' not in content:
    content = content.replace(
        'parser.add_argument("--config"',
        'parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])\n    parser.add_argument("--config"'
    )

# Configure logging
if 'Configure production-grade logging' not in content:
    content = content.replace(
        'args = parser.parse_args()',
        'args = parser.parse_args()\n    _logging.basicConfig(level=getattr(_logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")'
    )

# Replace prints with logger calls (Surgical replacement)
print_to_debug = [
    'print(f"  [Memory] {event:<30} | RSS: {rss_gb:.2f} GB")',
    'print(f"  [Memory] after_window_{win_idx} | RSS: {PreparedPanelCache.get_rss_mb() / 1024:.2f} GB")',
    'print("\\nFeature Polarity Sign-Flow Audit (Task 4):")',
    'print("\\nTarget Alignment Audit (Task 2):")',
    'print("--- FastSweep Performance Audit ---")'
]

for p in print_to_debug:
    if p in content:
        content = content.replace(p, f'logger.debug({p[6:-1]})')

# Replace some INFO prints
if 'print(f"Phase {phase_name} start")' in content:
    content = content.replace('print(f"Phase {phase_name} start")', 'logger.info(f"Phase {phase_name} start")')

path.write_text(content)
print("Patched logging")
