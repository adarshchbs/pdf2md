from app.pymupdf_parser.parameter.parsed_content import Block, Span
from typing import List, DefaultDict, Union, Tuple
from collections import defaultdict
import re


def get_block_no_for_page_desc_order(parsed_doc: List[Block]):
    parsed_doc_with_page: DefaultDict[
        Union[int, float], List[Tuple[int, Block]]
    ] = defaultdict(list)
    for block_no, block in enumerate(parsed_doc):
        parsed_doc_with_page[block.page_no].append((block_no, block))

    for page_no in range(len(parsed_doc_with_page)):
        parsed_content_with_block_no = parsed_doc_with_page[page_no]
        yield sorted(
            parsed_content_with_block_no,
            key=lambda x: x[1].block_bbox[3],
            reverse=True,
        )


def is_upper_and_smaller_than_next_span_text(
    span_first: Span, span_second: Span
) -> bool:
    is_upper = (span_second.start_span_bbox[3] - span_first.end_span_bbox[3]) > 0.05 * (
        span_second.start_span_bbox[3] - span_second.start_span_bbox[1]
    )
    is_smaller = (span_second.size - span_first.size) > span_second.size * 0.05
    return is_upper or is_smaller


def is_block_footnote(block: Block):
    if len(block.spans) == 1:
        return False

    regrex_match = bool(
        re.search(r"\A(\w\d|\d\w|\d{1,3}|\w)\Z", block.spans[0].text.strip())
    )
    is_superscript = is_upper_and_smaller_than_next_span_text(
        block.spans[0], block.spans[1]
    )
    return regrex_match and is_superscript


def filter_footnotes(
    parsed_doc: List[Block],
) -> tuple[list[Block], defaultdict[int, List[Tuple[str, str]]]]:
    footnotes: DefaultDict[int, List[Tuple[str, str]]] = defaultdict(list)
    for page_no, sorted_block_no in enumerate(
        get_block_no_for_page_desc_order(parsed_doc)
    ):
        footer_not_found_count = 0
        for block_no, block in sorted_block_no:
            if footer_not_found_count > 4:
                continue
            if is_block_footnote(block):
                footnotes_no = block.spans[0].text
                footnotes_text = "".join([span.text for span in block.spans[1:]])
                footnotes[page_no].append((footnotes_no, footnotes_text))
                print(f"{footnotes_no=}, {footnotes_text=}")
                parsed_doc[block_no] = None  # type: ignore
            else:
                footer_not_found_count += 1

    parsed_doc = list(filter(lambda x: x, parsed_doc))
    return parsed_doc, footnotes
