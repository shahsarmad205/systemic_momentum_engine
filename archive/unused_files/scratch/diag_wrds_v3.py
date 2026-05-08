import os
import pandas as pd
import sqlalchemy
import wrds
from sqlalchemy import text

# Import the new helper to test it
from utils.wrds_universe import wrds_query

def test_wrds_fix():
    wrds_user = "sarmadshah"
    
    print("--- Environment Diagnostics ---")
    print(f"Pandas version: {pd.__version__}")
    print(f"SQLAlchemy version: {sqlalchemy.__version__}")
    try:
        import wrds
        print("WRDS library loaded.")
    except Exception:
        print("WRDS library NOT found.")
    
    print(f"\nAttempting to connect to WRDS as {wrds_user}...")
    try:
        db = wrds.Connection(wrds_username=wrds_user)
        print("Connected successfully.")
        
        sql = "SELECT permno, start, ending FROM crsp.dsp500list LIMIT 5"
        print(f"Running query via new wrds_query helper: {sql}")
        
        try:
            df = wrds_query(db, sql)
            print("wrds_query SUCCESS!")
            print(f"Query row count: {len(df)}")
            print(df.head())
        except Exception as e:
            print(f"wrds_query FAILED: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Connection FAILED: {e}")

if __name__ == "__main__":
    test_wrds_fix()
