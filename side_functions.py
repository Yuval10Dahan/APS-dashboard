from sqlalchemy import create_engine, text
import pandas as pd





DB_PATH = "APS_data_base2.db"
TABLE = "Disruption Time Measurement"


DATES = ["09:45:35 25-01-2026",
         "11:44:27 12-01-2026",
         "19:14:44 11-01-2026",
         "18:58:05 11-01-2026",
         "17:20:22 11-01-2026",
         "12:20:44 11-01-2026",
         "16:48:51 10-08-2025",
         "20:43:51 10-08-2025",
         "14:45:22 08-12-2025"]


def delete_by_row_ids(db_path, table_name, row_ids):
    """
    Delete rows from SQLite table by rowid.
    row_ids: list[int]
    """
    if not row_ids:
        raise ValueError("row_ids list is empty")

    engine = create_engine(f"sqlite:///{db_path}")

    placeholders = ",".join(str(int(i)) for i in row_ids)
    query = text(f'''
        DELETE FROM "{table_name}"
        WHERE rowid IN ({placeholders})
    ''')

    with engine.begin() as conn:
        result = conn.execute(query)

    print(f"Deleted {result.rowcount} rows.")

def delete_by_timestamp(db_path, table_name, timestamp_value):


    """
    Delete rows where Time Stamp equals the given value.
    timestamp_value must match DB format exactly.
    """
    engine = create_engine(f"sqlite:///{db_path}")

    query = text(f'''
        DELETE FROM "{table_name}"
        WHERE "Time Stamp" = :ts
    ''')

    with engine.begin() as conn:
        result = conn.execute(query, {"ts": timestamp_value})

    print(f"Deleted {result.rowcount} rows.")

def preview_delete_by_ids(db_path, table_name, row_ids):
    engine = create_engine(f"sqlite:///{db_path}")
    ids = ",".join(str(i) for i in row_ids)
    df = pd.read_sql(
        f'SELECT rowid, * FROM "{table_name}" WHERE rowid IN ({ids})',
        engine
    )
    return df


if __name__ == '__main__':
    # id_list = list(range(1, 21))

    # preview_delete_by_ids(DB_PATH, TABLE, row_ids=id_list)


    # delete_by_row_ids(DB_PATH, TABLE, row_ids=id_list)

    for date in DATES:
        delete_by_timestamp(DB_PATH, TABLE, timestamp_value=date)
    