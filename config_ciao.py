import os
# Training parameters.
EPOCHS = 50
# Dataset name
DATASET_NAME = 'ciao'
# Data root.
DATA_ROOT_DIR = os.path.abspath('./data/' + DATASET_NAME)
# Number of parallel processes for data fetching.
NUM_WORKERS = 2
# For ASHA scheduler in Ray Tune.
MAX_NUM_EPOCHS = 50
GRACE_PERIOD = 2
# For search run (Ray Tune settings).
CPU = 2
GPU = 0.5
# Number of random search experiments to run.
NUM_SAMPLES = 100
# Test batch size
TEST_BATCH_SIZE = 1000
# Reduction factor
REDUCTION_FACTOR = 2
# Max failures
MAX_FAILURES = -1
# Local results dir
LOCAL_DIR='./outputs/' + DATASET_NAME + '/raytune_result'