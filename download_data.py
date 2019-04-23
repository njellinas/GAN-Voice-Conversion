import os
import wget
import zipfile

from speech_tools import *

dataset = 'vcc2018'

data_dir = os.path.join('datasets', dataset)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    wget.download(
        'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip?sequence=2&isAllowed=y',
        os.path.join(data_dir, 'train.zip'))
    wget.download(
        'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip?sequence=3&isAllowed=y',
        os.path.join(data_dir, 'eval_src.zip'))
    wget.download(
        'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip?sequence=5&isAllowed=y',
        os.path.join(data_dir, 'eval_trg.zip'))


def maybe_unzip(zip_filepath, destination_dir, force=False):
    print('Extracting zip file: ' + os.path.split(zip_filepath)[-1])
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(destination_dir)
    print("Extraction complete!")


maybe_unzip(os.path.join(data_dir, 'train.zip'), data_dir)
maybe_unzip(os.path.join(data_dir, 'eval_src.zip'), data_dir)
maybe_unzip(os.path.join(data_dir, 'eval_trg.zip'), data_dir)
