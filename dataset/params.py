import os

### DATA PARAMS
DATA_TITLE = "Title"
DATA_BODY = "Body"
DATA_SCORE = "Score"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

TRAIN_PATH = BASE_DIR + "/data/train.json"
TEST_PATH = BASE_DIR + "/data/test.json"
DEV_PATH = BASE_DIR + "/data/dev.json"

TOKENIZED_DATASET_PATH = BASE_DIR + "/data/tokenized.npy"

### MODEL PARAMS
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 768

BATCH_SIZE = 16

MODEL_SAVE_DIR = BASE_DIR + "/outputs/"

#TOKENIZER
PAD_TOKEN = '<|endoftext|>'
SEP_TOKEN = '*'
