import awkward as ak
import numpy as np
import pandas as pd

from app.pymupdf_parser.utils.cluster import moving_avg_cluster_1d, cluster_range
from app.utils.save_to_database import Connection


def put_text_with_their_center_in_database(
    unparsed_contents: ak.Array,
    con: Connection,
    table_name: str = "text_with_bbox",
):
    text_with_bbox_df = pd.DataFrame(columns=["text", "page_no", "o_x", "o_y"])
    for i, page in enumerate(unparsed_contents):
        text = ak.flatten(
            ak.flatten(
                page["blocks", :, "lines", :, "spans", ["bbox", "text", "origin"]]
            )
        )
        text["page_no"] = i
        bbox = np.array(text["bbox"].to_list())
        # origin_y = np.array(text["origin"].to_list())[:, 1]
        if len(bbox.shape) != 2:
            continue

        text["o_x"] = (bbox[:, 0] + bbox[:, 2]) / 2
        text["o_y"] = (bbox[:, 1] + bbox[:, 3]) / 2
        # text["y_index"] = moving_avg_cluster_1d(bbox[:, 3])
        text["y_index"] = cluster_range(bbox[:, [1, 3]])

        df = pd.DataFrame.from_records(text.to_list())
        df.drop(["bbox", "origin"], axis=1, inplace=True)
        text_with_bbox_df = pd.concat([text_with_bbox_df, df])

    print(f"{text_with_bbox_df.columns}")
    text_with_bbox_df.replace("", np.nan).dropna(axis=0, how="any")
    con.write_df_to_database(text_with_bbox_df, table_name)


def count_number_of_spans_inside_bbox(
    con: Connection, page_no: int, bbox: np.ndarray, table_name: str = "text_with_bbox"
):
    query = f"""--sql
                select count(*) 
                from {table_name}
                where page_no={page_no}
                and o_x between {bbox[0]} and {bbox[2]}
                and o_y between {bbox[1]} and {bbox[3]} 
                """
    return con.run_query(query).values[0][0]


def number_of_spans_per_line_inside_bbox(
    con: Connection, page_no: int, bbox: np.ndarray, table_name: str = "text_with_bbox"
):
    query = f"""--sql
                select count(*)
                from {table_name}
                where page_no={page_no}
                and o_x between {bbox[0]} and {bbox[2]}
                and o_y between {bbox[1]} and {bbox[3]} 
                group by y_index
                order by y_index
                """
    return con.run_query(query).values[:, 0]
