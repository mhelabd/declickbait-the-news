import os

### DATA PARAMS
DATA_TITLE = "Title"
DATA_BODY = "Body"
DATA_SCORE = "Score"
DATA_SUMMARY = "Summary"


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

TRAIN_PATH_T5 = BASE_DIR + "/dataset/data/train.json"
TEST_PATH = BASE_DIR + "/dataset/data/test.json"
DEV_PATH = BASE_DIR + "/dataset/data/dev.json"


TOKENIZED_DATASET_PATH = BASE_DIR + "/dataset/data/tokenized_train.pt"
TOKENIZED_DATASET_PATH_TRAIN = BASE_DIR + "/dataset/data/tokenized_train.pt"
TOKENIZED_DATASET_PATH_DEV = BASE_DIR + "/dataset/data/tokenized_dev.pt"
TOKENIZED_DATASET_PATH_TEST = BASE_DIR + "/dataset/data/tokenized_test.pt"

TOKENIZED_DATASET_PATH_T5 = BASE_DIR + "/dataset/data/tokenized_t5.pt"
TOKENIZED_DATASET_PATH_TRAIN_T5 = BASE_DIR + "/dataset/data/tokenized_train_t5.pt"
TOKENIZED_DATASET_PATH_DEV_T5 = BASE_DIR + "/dataset/data/tokenized_dev_t5.pt"
TOKENIZED_DATASET_PATH_TEST_T5 = BASE_DIR + "/dataset/data/tokenized_test_t5.pt"

METRICS_PATH_TRAIN = BASE_DIR + "/metrics/train_metric_outputs.json"
METRICS_PATH_DEV = BASE_DIR + "/metrics/dev_metric_outputs.json"
METRICS_PATH_TEST = BASE_DIR + "/metrics/test_metric_outputs.json"

OUTPUT_PATH_TRAIN = BASE_DIR + "/metrics/train_outputs.json"
OUTPUT_PATH_DEV = BASE_DIR + "/metrics/dev_outputs.json"
OUTPUT_PATH_TEST = BASE_DIR + "/metrics/test_outputs.json"

### MODEL PARAMS
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 512

MODEL_SAVE_DIR = BASE_DIR + "/outputs/"

#TOKENIZER
PAD_TOKEN = '<|endoftext|>'
SEP_TOKEN = '*'

