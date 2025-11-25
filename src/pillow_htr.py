
import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
import tensorflow as tf
import PIL
from PIL import Image
import numpy as np
# Disable eager mode
tf.compat.v1.disable_eager_execution()

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

model = None    

def init_model():
    global model
    model = Model(char_list_from_file(), DecoderType.BestPath, must_restore=True, dump=False)

def infer(pil_image : PIL.Image) -> None:
    global model
    if model is None:
        model = Model(char_list_from_file(), DecoderType.BestPath, must_restore=True, dump=False)
    """Recognizes text in image provided by file path."""
    #img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    img = None
    img_c = np.array(pil_image)
    img= cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    return recognized[0]
import os
def main() :
    os.chdir(__file__.removesuffix("pillow_htr.py"))
    im = Image.open("../data/word.png")
    infer(im)

if __name__ == '__main__':
    main()