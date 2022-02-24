import sys
import os

### DATA PARAMS
DATA_TITLE = "Title"
DATA_BODY = "Body"
DATA_SCORE = "Score"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

TRAIN_PATH = BASE_DIR + "/dataset/data/train.json"
print(TRAIN_PATH)
TEST_PATH = BASE_DIR, "/dataset/data/test.json"
DEV_PATH = BASE_DIR + "/dataset/data/dev.json"

### MODEL PARAMS
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 1024

