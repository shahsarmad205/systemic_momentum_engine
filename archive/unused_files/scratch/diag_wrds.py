import os
import pandas as pd
import numpy as np

# We'll try to simulate what happens in connect_wrds and raw_sql
def test_wrds_connection():
    wrds_user = os.environ.get("WRDS_USERNAME")
    if not wrds_user or wrds_user == "sarmadshah": # Use the user's provided username
        wrds_user = "sarmadshah"
    
    print(f"Attempting to connect to WRDS as {wrds_user}...")
    try:
        import wrds
        db = wrds.Connection(wrds_username=wrds_user)
        print("Connected successfully.")
        print(f"Connection type: {type(db)}")
        
        # Check internal attributes
        if hasattr(db, 'connection'):
            print(f"db.connection type: {type(db.connection)}")
        if hasattr(db, 'engine'):
            print(f"db.engine type: {type(db.engine)}")
            
        sql = "SELECT permno, start, ending FROM crsp.dsp500list LIMIT 5"
        print(f"Running test query: {sql}")
        
        # Try raw_sql
        try:
            df = db.raw_sql(sql)
            print("db.raw_sql worked!")
            print(df.head())
        except Exception as e:
            print(f"db.raw_sql FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            # Try manual fix if it's an SQLAlchemy 2.0 issue
            print("\nTrying manual fix with sqlalchemy.text and db.engine...")
            try:
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    df = pd.read_sql(text(sql), conn)
                print("Manual fix with text() and engine.connect() worked!")
                print(df.head())
            except Exception as e2:
                print(f"Manual fix FAILED: {e2}")

    except Exception as e:
        print(f"Connection FAILED: {e}")

if __name__ == "__main__":
    test_wrds_connection()
