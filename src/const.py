
INPUT_DATA_DIR = '../data/input/'
TRAIN_PATH = f'{INPUT_DATA_DIR}/train.csv'
TEST_PATH = f'{INPUT_DATA_DIR}/test.csv'
SAMPLE_PATH = f'{INPUT_DATA_DIR}sample_submission.csv'

OUTPUT_DATA_DIR = '../data/output'

TARGET_COL = 'label'
PIXEL_COLS = [f'pixel{i}' for i in range(28 ** 2)]
