import os


MODEL_STORE_DIR = "./models"

ANNO_STORE_DIR = "./annotations"

TRAIN_DATA_DIR = "./data"

LOG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/log"

USE_CUDA = True

TRAIN_BATCH_SIZE = 512

TRAIN_LR = 0.01

END_EPOCH = 10

NUM_LANDMARKS = 5

PNET_SIZE = 12
PNET_POSTIVE_ANNO_FILENAME = "{0}_landmarks_pos_{1}.txt".format(NUM_LANDMARKS, PNET_SIZE)
PNET_NEGATIVE_ANNO_FILENAME = "{0}_landmarks_neg_{1}.txt".format(NUM_LANDMARKS, PNET_SIZE)
PNET_PART_ANNO_FILENAME = "{0}_landmarks_part_{1}.txt".format(NUM_LANDMARKS, PNET_SIZE)
PNET_LANDMARK_ANNO_FILENAME = "{0}_landmarks_landmark_{1}.txt".format(NUM_LANDMARKS, PNET_SIZE)

RNET_SIZE = 24
RNET_POSTIVE_ANNO_FILENAME = "{0}_landmarks_pos_{1}.txt".format(NUM_LANDMARKS, RNET_SIZE)
RNET_NEGATIVE_ANNO_FILENAME = "{0}_landmarks_neg_{1}.txt".format(NUM_LANDMARKS, RNET_SIZE)
RNET_PART_ANNO_FILENAME = "{0}_landmarks_part_{1}.txt".format(NUM_LANDMARKS, RNET_SIZE)
RNET_LANDMARK_ANNO_FILENAME = "{0}_landmarks_landmark_{1}.txt".format(NUM_LANDMARKS, RNET_SIZE)

ONET_SIZE = 48
ONET_POSTIVE_ANNO_FILENAME = "{0}_landmarks_pos_{1}.txt".format(NUM_LANDMARKS, ONET_SIZE)
ONET_NEGATIVE_ANNO_FILENAME = "{0}_landmarks_neg_{1}.txt".format(NUM_LANDMARKS, ONET_SIZE)
ONET_PART_ANNO_FILENAME = "{0}_landmarks_part_{1}.txt".format(NUM_LANDMARKS, ONET_SIZE)
ONET_LANDMARK_ANNO_FILENAME = "{0}_landmarks_landmark_{1}.txt".format(NUM_LANDMARKS, ONET_SIZE)

PNET_TRAIN_IMGLIST_FILENAME = "{0}_landmarks_imglist_anno_{1}.txt".format(NUM_LANDMARKS, PNET_SIZE)
RNET_TRAIN_IMGLIST_FILENAME = "{0}_landmarks_imglist_anno_{1}.txt".format(NUM_LANDMARKS, RNET_SIZE)
ONET_TRAIN_IMGLIST_FILENAME = "{0}_landmarks_imglist_anno_{1}.txt".format(NUM_LANDMARKS, ONET_SIZE)

BASE_DATA_PATH = '/dev/data/img_wider'
BASE_DATA_ANNO_FILE = 'wider_anno.txt'
BASE_LANDMARK_PATH = '/dev/data/img_celeba'
BASE_LANDMARK_ANNO_FILE = 'celeba_anno.txt'
