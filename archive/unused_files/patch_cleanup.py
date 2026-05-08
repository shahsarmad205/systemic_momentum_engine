import sys
from pathlib import Path

path = Path("/Users/sarmad/Desktop/Quant-project/trend_signal_engine/run_model_selection.py")
content = path.read_text()

# Find the return in _evaluate_model_family
# It looks like:
#     from model_selection import validation as _val
#     return {
#         "model_name": name,

target = "    from model_selection import validation as _val\n    return {"
replacement = """    # Final cleanup of reloaded objects and scratch files
    try:
        del oos_df
        del pnl_parts
        del daily_parts
        del overlay_pnl_parts
        del overlay_daily_parts
    except NameError:
        pass
        
    try:
        import shutil
        shutil.rmtree(scratch_dir, ignore_errors=True)
    except:
        pass

    import gc
    gc.collect()

    from model_selection import validation as _val
    return {"""

if target in content:
    new_content = content.replace(target, replacement)
    path.write_text(new_content)
    print("Successfully patched run_model_selection.py")
else:
    print("Could not find target string in run_model_selection.py")
    sys.exit(1)
