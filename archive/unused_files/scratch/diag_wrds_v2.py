import os
import pandas as pd
from sqlalchemy import text
import wrds

def test_fixed_query():
    wrds_user = "sarmadshah"
    db = wrds.Connection(wrds_username=wrds_user)
    sql = "SELECT permno, start, ending FROM crsp.dsp500list LIMIT 5"
    
    print("\nAttempt 1: pd.read_sql(text(sql), db.engine)")
    try:
        df = pd.read_sql(text(sql), db.engine)
        print("Success!")
        print(df.head())
    except Exception as e:
        print(f"Attempt 1 Failed: {e}")

    print("\nAttempt 2: pd.read_sql(sql, db.engine) (string query)")
    try:
        df = pd.read_sql(sql, db.engine)
        print("Success!")
        print(df.head())
    except Exception as e:
        print(f"Attempt 2 Failed: {e}")

    print("\nAttempt 3: connection.execute(text(sql))")
    try:
        with db.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        print("Success!")
        print(df.head())
    except Exception as e:
        print(f"Attempt 3 Failed: {e}")

if __name__ == "__main__":
    test_fixed_query()
