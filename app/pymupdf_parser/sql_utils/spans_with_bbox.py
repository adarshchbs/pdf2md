import awkward as ak
import numpy as np
import polars as pl

from app.pymupdf_parser.utils.cluster import cluster_range
from app.utils.save_to_database import Connection, _quoted_identifier


def _bbox_parameters(bbox: np.ndarray) -> tuple[object, object, object, object]:
    if not isinstance(bbox, np.ndarray) or bbox.shape != (4,):
        raise ValueError("bbox must be a one-dimensional NumPy array with four coordinates")
    return bbox[0], bbox[2], bbox[1], bbox[3]


def put_text_with_their_center_in_database(
    unparsed_contents: ak.Array,
    con: Connection,
    table_name: str = "text_with_bbox",
):
    page_frames: list[pl.DataFrame] = []
    for i, page in enumerate(unparsed_contents):
        text = ak.flatten(ak.flatten(page["blocks", :, "lines", :, "spans", ["bbox", "text", "origin"]]))
        text["page_no"] = i
        bbox = np.array(text["bbox"].to_list())
        if len(bbox.shape) != 2:
            continue

        text["o_x"] = (bbox[:, 0] + bbox[:, 2]) / 2
        text["o_y"] = (bbox[:, 1] + bbox[:, 3]) / 2
        text["y_index"] = cluster_range(bbox[:, [1, 3]])

        page_frames.append(pl.from_dicts(text.to_list()).drop("bbox", "origin"))

    if page_frames:
        text_with_bbox_df = pl.concat(page_frames, how="vertical_relaxed")
    else:
        text_with_bbox_df = pl.DataFrame(
            schema={
                "text": pl.String,
                "page_no": pl.Int64,
                "o_x": pl.Float64,
                "o_y": pl.Float64,
                "y_index": pl.Int32,
            }
        )

    print(f"{text_with_bbox_df.columns}")
    con.write_df_to_database(text_with_bbox_df, table_name)


def count_number_of_spans_inside_bbox(
    con: Connection, page_no: int, bbox: np.ndarray, table_name: str = "text_with_bbox"
):
    query = f"""--sql
                select count(*)
                from {_quoted_identifier(table_name)}
                where page_no=?
                and o_x between ? and ?
                and o_y between ? and ?
                """
    result = con.connection.execute(query, [page_no, *_bbox_parameters(bbox)]).pl()
    return result.item(0, 0)


def number_of_spans_per_line_inside_bbox(
    con: Connection, page_no: int, bbox: np.ndarray, table_name: str = "text_with_bbox"
):
    query = f"""--sql
                select count(*)
                from {_quoted_identifier(table_name)}
                where page_no=?
                and o_x between ? and ?
                and o_y between ? and ?
                group by y_index
                order by y_index
                """
    result = con.connection.execute(query, [page_no, *_bbox_parameters(bbox)]).pl()
    return result.to_numpy()[:, 0]
