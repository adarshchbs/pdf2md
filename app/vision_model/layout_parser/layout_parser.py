"""Copyright (C) 2022 Adarsh Gupta"""
import fitz
import layoutparser as lp
import numpy as np
from PIL import Image
from tqdm import tqdm

from app.utils.pix2np import pix2np

model = lp.AutoLayoutModel(
    "lp://efficientdet/PubLayNet",
    label_map={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
)

# model = lp.AutoLayoutModel(
#     "lp://efficientdet/MFD",
#     label_map={1: 3},
# )

# model = lp.Detectron2LayoutModel(
#     config_path="lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
#     label_map={1: 0, 2: 4, 3: 3, 4: 5, 5: 6, 6: 7},
# )

# model = lp.Detectron2LayoutModel(
#     config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
#     label_map={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
# )


def layoutparser(file_path: str):
    document = fitz.open(file_path)
    document_layout_dict = {"pred_boxes": {}, "pred_classes": {}}
    for i, page in enumerate(tqdm(document)):
        img = pix2np(page.get_pixmap())
        # page_width, page_height = page.rect[2], page.rect[3]
        # img_height, imag_width = img.shape[0], img.shape[1]
        output = model.detect(img)
        pred_boxes = np.array(
            [
                [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2]
                for block in output
            ]
        )
        pred_classes = np.array([block.type for block in output])
        document_layout_dict["pred_boxes"][i] = pred_boxes
        document_layout_dict["pred_classes"][i] = pred_classes
        for block in output:
            # print(block.type)
            # if block.type in {3, 4}:
            if block.type in {3, 4}:
                rect = block.block
                page.add_highlight_annot([rect.x_1, rect.y_1, rect.x_2, rect.y_2])

    document.save("pdf/table_n_figure.pdf")
    return document_layout_dict
