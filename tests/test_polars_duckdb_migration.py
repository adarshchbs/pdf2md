from pathlib import Path

import awkward as ak
import duckdb
import numpy as np
import polars as pl
import pytest

from app.pymupdf_parser.alignment import horizontal_margin
from app.pymupdf_parser.sql_utils.spans_with_bbox import (
    count_number_of_spans_inside_bbox,
    number_of_spans_per_line_inside_bbox,
)
from app.pymupdf_parser.utils.approximate_numbers import ApproximateNumbersWithThreshold
from app.utils.save_to_database import Connection, write_to_database


def test_duckdb_round_trip_preserves_schema_order_and_nulls() -> None:
    connection = Connection(":memory:")
    source = pl.DataFrame({
        "row_id": pl.Series([2, 1], dtype=pl.Int64),
        "label": pl.Series([None, "kept"], dtype=pl.String),
        "score": pl.Series([2.5, 1.5], dtype=pl.Float64),
    })

    connection.write_df_to_database(source, "event data")
    actual = connection.run_query('SELECT * FROM "event data" ORDER BY row_id')

    expected = pl.DataFrame({
        "row_id": pl.Series([1, 2], dtype=pl.Int64),
        "label": pl.Series(["kept", None], dtype=pl.String),
        "score": pl.Series([1.5, 2.5], dtype=pl.Float64),
    })
    assert actual.equals(expected)


def test_duckdb_quoted_identifiers_and_typed_empty_frames() -> None:
    connection = Connection(":memory:")
    source = pl.DataFrame(
        schema={
            'row" id': pl.UInt32,
            "nullable": pl.String,
        }
    )

    connection.write_df_to_database(source, 'event" data')
    actual = connection.run_query('SELECT * FROM "event"" data"')

    assert actual.columns == ['row" id', "nullable"]
    assert actual.schema == source.schema
    assert actual.is_empty()


def test_duckdb_replacement_participates_in_transactions() -> None:
    connection = Connection(":memory:")
    original = pl.DataFrame({"old_column": [1]})
    replacement = pl.DataFrame({"new_column": [2]})
    connection.write_df_to_database(original, "replace_me")

    connection.connection.begin()
    connection.write_df_to_database(replacement, "replace_me")
    connection.connection.rollback()

    assert connection.run_query("SELECT * FROM replace_me").equals(original)


def test_connection_context_and_convenience_writer_close_connections(tmp_path: Path) -> None:
    database = tmp_path / "lifecycle.duckdb"
    with Connection(str(database)) as connection:
        connection.write_df_to_database(pl.DataFrame({"value": [1]}), "first")
    with pytest.raises(duckdb.ConnectionException):
        connection.run_query("SELECT 1")

    write_to_database(pl.DataFrame({"value": [2]}), str(database), "second")
    with Connection(str(database)) as reader:
        assert reader.run_query("SELECT value FROM second").item() == 2


def test_duckdb_rejects_lazy_frames() -> None:
    connection = Connection(":memory:")
    with pytest.raises(AssertionError):
        connection.write_df_to_database(pl.DataFrame({"value": [1]}).lazy(), "lazy")  # type: ignore[arg-type]


def test_duckdb_array_write_preserves_columns_and_shape_boundary() -> None:
    connection = Connection(":memory:")
    connection.write_array_to_database(
        np.array([[2, 20], [1, 10]], dtype=np.int64),
        "measurements",
        ["row_id", "value"],
    )

    actual = connection.run_query("SELECT * FROM measurements ORDER BY row_id")
    expected = pl.DataFrame(
        {"row_id": [1, 2], "value": [10, 20]}, schema={"row_id": pl.Int64, "value": pl.Int64}
    )
    assert actual.equals(expected)

    with pytest.raises(AssertionError):
        connection.write_array_to_database(
            np.array([[1, 2]], dtype=np.int64),
            "invalid",
            ["only_one_column"],
        )
    with pytest.raises(AssertionError):
        connection.write_array_to_database(
            np.array([1, 2], dtype=np.int64),
            "one_dimensional",
            ["value"],
        )
    with pytest.raises(AssertionError):
        connection.write_array_to_database(
            [np.array([1, 2]), np.array([3])],
            "ragged",
            ["left", "right"],
        )


def test_duckdb_bbox_queries_preserve_group_order_and_inclusive_bounds() -> None:
    connection = Connection(":memory:")
    connection.write_df_to_database(
        pl.DataFrame({
            "text": ["a", "b", "c", "outside"],
            "page_no": [0, 0, 0, 1],
            "o_x": [0.0, 1.0, 2.0, 1.0],
            "o_y": [0.0, 1.0, 2.0, 1.0],
            "y_index": [1, 0, 1, 0],
        }),
        "spans",
    )

    bbox = np.array([0.0, 0.0, 2.0, 2.0])
    assert count_number_of_spans_inside_bbox(connection, 0, bbox, "spans") == 3
    np.testing.assert_array_equal(
        number_of_spans_per_line_inside_bbox(connection, 0, bbox, "spans"),
        np.array([1, 2]),
    )
    np.testing.assert_array_equal(
        number_of_spans_per_line_inside_bbox(
            connection,
            0,
            np.array([10.0, 10.0, 20.0, 20.0]),
            "spans",
        ),
        np.array([], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="four coordinates"):
        count_number_of_spans_inside_bbox(connection, 0, np.array([0.0, 1.0, 2.0]), "spans")


def test_duckdb_bbox_query_quotes_table_names() -> None:
    connection = Connection(":memory:")
    connection.write_df_to_database(
        pl.DataFrame({
            "page_no": [0],
            "o_x": [1.0],
            "o_y": [1.0],
            "y_index": [0],
        }),
        'span" data',
    )

    assert (
        count_number_of_spans_inside_bbox(
            connection,
            0,
            np.array([0.0, 0.0, 2.0, 2.0]),
            'span" data',
        )
        == 1
    )


def test_horizontal_margin_polars_pipeline_preserves_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate_data = pl.DataFrame({
        "x_0": [10.0, 10.0, 20.0, 20.0],
        "x_1": [90.0, 90.0, 80.0, 80.0],
        "page_no": [0, 1, 0, 1],
        "no_of_lines": [3, 3, 4, 4],
    })
    monkeypatch.setattr(
        horizontal_margin,
        "put_cordinate_information_df",
        lambda _document: coordinate_data,
    )

    result = horizontal_margin.horizontal_margin_per_page(ak.Array([{}, {}]))

    assert list(result) == [0, 1]
    expected = np.array([[20.0, 80.0], [10.0, 90.0]])
    np.testing.assert_array_equal(result[0], expected)
    np.testing.assert_array_equal(result[1], expected)


def test_approximate_numbers_uses_cluster_means() -> None:
    values = np.array([1.0, 1.2, 5.0])
    approximate = ApproximateNumbersWithThreshold(values, threshold=0.5)

    assert [approximate.approx(value) for value in values] == [1.1, 1.1, 5.0]
