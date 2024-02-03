import sqlite3 as sl
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class Connection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = sl.connect(db_path)

    def write_df_to_database(self, df: pd.DataFrame, table_name: str):
        assert isinstance(df, pd.DataFrame)
        with self.connection as con:
            con.execute(f"drop table if exists {table_name}")
        df.to_sql(table_name, self.connection)
        print(f"Created table {table_name} at {self.db_path}")

    def write_array_to_database(
        self,
        array: Union[np.ndarray, List[np.ndarray]],
        table_name: str,
        columns: List[str],
    ):
        if isinstance(array, np.ndarray):
            assert len(array.shape) == 2
            assert array.shape[1] == len(columns)
        elif isinstance(array, list):
            for row in array:
                assert len(columns) == len(row)
        else:
            raise ValueError("array is of not type ndarray or list of ndarray")

        df = pd.DataFrame(array, columns=columns)
        self.write_df_to_database(df, table_name)

    def run_query(self, table_name: str):
        return pd.read_sql_query(table_name, self.connection)


def write_to_database(
    array: Union[pd.DataFrame, np.ndarray, List[np.ndarray]],
    db_path: str,
    table_name: str,
    columns: Optional[List[str]] = None,
):  # sourcery skip: instance-method-first-arg-name
    if columns is None:
        columns = []
    connection = Connection(db_path)
    if isinstance(array, (np.ndarray, list)):
        assert columns is not None
        connection.write_array_to_database(array, table_name, columns)
    elif isinstance(array, pd.DataFrame):
        connection.write_df_to_database(array, table_name)
    else:
        raise ValueError("array is of not type ndarray or list of ndarray")
