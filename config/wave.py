# DATA_DIR = '/Users/vcerqueira/Downloads/IWaveBNetwork30Min_fc98_2d2c_3a8f.csv'
DATA_DIR = '../data/wave_data.csv'

UNUSED_COLUMNS = ['latitude', 'longitude', 'station_id',
                  'Hmax', 'THmax', 'MeanCurDirTo',
                  'MeanCurSpeed', 'SeaTemperature', 'PeakDirection']

BUOY_ID = 'AMETS Berth B Wave Buoy'
EMBED_DIM = 5
MAX_HORIZON = 24
TARGET = 'SignificantWaveHeight'
THRESHOLD_PERCENTILE = 0.95
CV_N_FOLDS = 10
TRAIN_SIZE = 0.5
TEST_SIZE = 0.2
