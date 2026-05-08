import sys
from unittest.mock import MagicMock
import pandas as pd

# Mock wrds before any imports
mock_wrds = MagicMock()
mock_conn = MagicMock()
# Mock raw_sql to return empty DF by default
mock_conn.raw_sql.return_value = pd.DataFrame()
mock_wrds.Connection.return_value = mock_conn
sys.modules["wrds"] = mock_wrds

from run_model_selection import main

if __name__ == "__main__":
    # main() in run_model_selection.py calls parser.parse_args() which uses sys.argv[1:]
    # So we keep sys.argv as is.
    main()
