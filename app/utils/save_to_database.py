from typing import List, Optional, Union

import duckdb
import numpy as np
import polars as pl


def _quoted_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


class Connection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = duckdb.connect(db_path)

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def write_df_to_database(self, df: pl.DataFrame, table_name: str):
        if not isinstance(df, pl.DataFrame):
            raise AssertionError("df must be an eager Polars DataFrame")
        registered_name = "_pdf2md_dataframe"
        self.connection.register(registered_name, df)
        try:
            self.connection.execute(
                f"CREATE OR REPLACE TABLE {_quoted_identifier(table_name)} AS SELECT * FROM {registered_name}"
            )
        finally:
            self.connection.unregister(registered_name)
        print(f"Created table {table_name} at {self.db_path}")

    def write_array_to_database(
        self,
        array: Union[np.ndarray, List[np.ndarray]],
        table_name: str,
        columns: List[str],
    ):
        if isinstance(array, np.ndarray):
            if array.ndim != 2:
                raise AssertionError("array must be two-dimensional")
            if array.shape[1] != len(columns):
                raise AssertionError("array width must match the number of columns")
        elif isinstance(array, list):
            for row in array:
                if len(columns) != len(row):
                    raise AssertionError("every row width must match the number of columns")
        else:
            raise ValueError("array is of not type ndarray or list of ndarray")

        df = pl.DataFrame(array, schema=columns, orient="row")
        self.write_df_to_database(df, table_name)

    def run_query(self, query: str) -> pl.DataFrame:
        return self.connection.execute(query).pl()


def write_to_database(
    array: Union[pl.DataFrame, np.ndarray, List[np.ndarray]],
    db_path: str,
    table_name: str,
    columns: Optional[List[str]] = None,
):  # sourcery skip: instance-method-first-arg-name
    if columns is None:
        columns = []
    with Connection(db_path) as connection:
        if isinstance(array, (np.ndarray, list)):
            connection.write_array_to_database(array, table_name, columns)
        elif isinstance(array, pl.DataFrame):
            connection.write_df_to_database(array, table_name)
        else:
            raise ValueError("array is of not type ndarray or list of ndarray")
