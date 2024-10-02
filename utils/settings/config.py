import numpy as np
import importlib.util
import sys
from pathlib import Path


def load_module(module_file, module_name):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


AUTHORS = 'Papoutsis I., Sykas D., Zografakis D., Sdraka M.'
DATASET_VERSION = '21.03'
RANDOM_SEED = 42
LICENSES = [
    {
        'url': 'url',
        'id': 1,
        'name': 'name'
    },
]

# Divisors of the Positive Integer 10980:
# 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 61, 90, 122, 180, 183,
# 244, 305, 366, 549, 610, 732, 915, 1098, 1220, 1830, 2196, 2745, 3660, 5490, 10980
IMG_SIZE = 366

# Total pixels for each resolution for Sentinel2 Data
SENTINEL2_PIXELS = {
    '10': 10980,
    '20': 5490,
    '60': 1830,
}

# Band names and their resolutions
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

# Extract patches based on this band
REFERENCE_BAND = 'B02'

# File to load Mappings from (aka native str to english str)
# You can replace the path to your own mappings, be sure to name mapping dictionary as 'CLASSES_MAPPING'
MAPPINGS_FILE = Path('utils/settings/mappings/mappings_cat.py')
#MAPPINGS_FILE = Path('utils/settings/mappings/mappings_fr.py')
module = load_module(MAPPINGS_FILE, MAPPINGS_FILE.stem)
CLASSES_MAPPING = module.CLASSES_MAPPING
RENAME = module.RENAME
SAMPLE_TILES = module.SAMPLE_TILES
COUNTRY_CODE = module.COUNTRY_CODE

# File to load Encodings from (aka english str to ints)
ENCODINGS_FILE = Path('utils/settings/mappings/encodings_en.py')
module = load_module(ENCODINGS_FILE, ENCODINGS_FILE.stem)
CROP_ENCODING = module.CROP_ENCODING

'''
        Label  Count
1        CTC4  15702
2    RB966928   5875
6     CTC9003   4791
8     CTC9001   4372
0     CTC9002   3130
5    RB975242   2007
4      CV7870   1797
12   RB867515   1745
7    RB975201   1404
10   RB985476   1310
13    CTC2994   1297
9       CTC20   1229
3    RB975033   1169
11  CTC9005HP   1150
'''

# --- For the selected classes, uncomment this section ---
SELECTED_CLASSES = [
    "CTC4",
    "RB966928",
    "CTC9003",
    "CTC9001",
    "CTC9002",
    "RB975242",
    "CV7870",
    "RB867515",
    "RB975201",
    "RB985476",
    "CTC2994",
    "CTC20",
    "RB975033",
    "CTC9005HP"
]

LINEAR_ENCODER = {val: i  for i, val in enumerate(sorted(SELECTED_CLASSES))}
# --- Binary classification ---

SELECTED_CLASSES_BINARY = [
    'CTC', 'else'
    ]

LINEAR_ENCODER_BINARY = {val: i  for i, val in enumerate(sorted(SELECTED_CLASSES_BINARY))}
# --- CTC Classes ---
SELECTED_CLASSES_CTC = [
    'CTC4', 'CTC9003', 'CTC9001', 'CTC9002', 'CTC2994', 'CTC20', 'CTC9005HP'
]
LINEAR_ENCODER_CTC = {val: i  for i, val in enumerate(sorted(SELECTED_CLASSES_CTC))}
# Class weights for loss function

# --- Uncomment only the weights of the experiment to be run ---
# Experiment 1
# ------------
CLASS_WEIGHTS_EQUAL = {0: 1.3, 
                 1: 1, 
                 2: 1.8, 
                 3: 1.5, 
                 4: 1., 
                 5: 1., 
                 6: 1.5, 
                 7: 1., 
                 8: 1., 
                 9: 1., 
                 10: 1., 
                 11: 1.,
                 12: 1.,
                 13: 1.3
                 }
CLASS_WEIGHT_PIXEL = {0: 3.6966489080683944,
 1: 2.170869920310061,
 2: 0.23278630654884763,
 3: 0.7643338242162675,
 4: 0.9260760150238141,
 5: 0.612661888518328,
 6: 3.4687029455415344,
 7: 1.8114072157631909,
 8: 1.9796426423051752,
 9: 0.5932712730200347,
 10: 2.319699169845929,
 11: 2.2410011808354446,
 12: 1.5561299092431107,
 13: 2.4895482683898713}
CLASS_WEIGHT_SAMPLE = {0: 2.739435589998538,
 1: 2.5959539975058887,
 2: 0.2135334746632018,
 3: 0.7686469188479528,
 4: 1.0701433712229393,
 5: 0.7006357516828721,
 6: 2.9250585480093676,
 7: 1.8703204552261157,
 8: 1.921341400881961,
 9: 0.5709105314480741,
 10: 2.8747890133497007,
 11: 2.3918039065492147,
 12: 1.6717230302489516,
 13: 2.556283258288989}


# weights with number of sample cap of 2000
CLASS_WEIGHT_SAMPLE_2000 = {0: 1.494736072525223,
 1: 1.4164472772620202,
 2: 0.7301785714285715,
 3: 0.7301785714285715,
 4: 0.7301785714285715,
 5: 0.7301785714285715,
 6: 1.5960187353629978,
 7: 1.0205151242887092,
 8: 1.0483540149728232,
 9: 0.7301785714285715,
 10: 1.5685898419518183,
 11: 1.3050555342780543,
 12: 0.9121531185865976,
 13: 1.3948014735980352}


# class weights for shuffled pixels
CLASS_WEIGHT_PIXEL_SHUFFLED_CAP_2000 = {0: 2.071968399127402,
 1: 1.2185678816427594,
 2: 2.320386584433942,
 3: 1.7483838236049155,
 4: 1.2509027641368174,
 5: 0.6582927094139766,
 6: 1.9455861487385744,
 7: 1.0167905347374941,
 8: 1.110414169551651,
 9: 2.7814665306839639,
 10: 1.299375586450463,
 11: 1.2577079954274224,
 12: 1.8731782524312186,
 13: 1.973839429098973}