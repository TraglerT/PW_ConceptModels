from pandas import DataFrame
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class Derm7pt_data(object):
    #multiple specific class names grouped together by more general names
    diagnosis: DataFrame = pd.DataFrame([
        {'nums': 0, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'nums': 1,
         'names': ['nevus', 'blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus',
                   'recurrent nevus', 'reed or spitz nevus'], 'abbrevs': 'NEV', 'info': 'Moles'},
        {'nums': 2,
         'names': ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                   'melanoma (0.76 to 1.5 mm)',
                   'melanoma (more than 1.5 mm)', 'melanoma metastasis'], 'abbrevs': 'MEL', 'info': 'Melanoma, cancer'},
        {'nums': 3, 'names': ['DF/LT/MLS/MISC', 'dermatofibroma', 'lentigo', 'melanosis',
                              'miscellaneous', 'vascular lesion'], 'abbrevs': 'MISC', 'info': 'benign'},
        {'nums': 4, 'names': 'seborrheic keratosis', 'abbrevs': 'SK', 'info': 'benign'},
    ])

    def __init__(self, data_folder: str):
        self.images = self.loadImages(os.path.join(data_folder, "images"))
        self.metadata = self.loadMeta(os.path.join(data_folder, "meta\\meta.csv"))
        self.labels = self.metadata['nums']

    def loadImages(self, input_dir: str):
        """
        Load all images from a directory

        :param input_dir:
        :return: {tail\filename: PixelImage}
        """
        result = {}
        for dir, subdirs, files in os.walk(input_dir):
            head, tail = os.path.split(dir)
            for file in files:
                if file.endswith(tuple(['.jpg', '.JPG', '.jpeg', '.JPEG'])):
                    img = Image.open(os.path.join(dir, file))
                    key = os.path.normpath(os.path.join(tail, file)).upper()
                    result[key] = img
        return result

    def loadImages_ToDo(self):
        #ToDo: add both clinical and dermoscopic images into dataset
        pass


    #load meta data
    def loadMeta(self, file_path: str):
        df = pd.read_csv(file_path)
        #drop columns that are not needed
        drop_columns = ['case_id', 'notes', 'management']
        df = df.drop(drop_columns, axis=1)
        #diagnosis is very specific, so we merge it with the more general diagnosis grouping.
        merged_df = df.merge(self.diagnosis.explode('names'), how='left', left_on='diagnosis', right_on='names')
        merged_df = merged_df.drop('names', axis=1)
        return merged_df



class Derm7PtDatasetGroupInfrequent(object):

    vascular_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'arborizing', 'comma', 'hairpin', 'within regression', 'wreath'],
         'abbrevs': 'REG', 'scores': 0, 'info': ''},
        {'nums': 2, 'names': ['dotted/irregular', 'dotted', 'linear irregular'], 'abbrevs': 'IR', 'scores': 2,
         'info': ''},
    ])

    pigmentation = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['regular', 'diffuse regular', 'localized regular'], 'abbrevs': 'REG', 'scores': 0,
         'info': ''},
        {'nums': 2, 'names': ['irregular', 'diffuse irregular', 'localized irregular'], 'abbrevs': 'IR', 'scores': 1,
         'info': ''},
    ])

    regression_structures = pd.DataFrame([
        {'nums': 0, 'names': 'absent', 'abbrevs': 'ABS', 'scores': 0, 'info': ''},
        {'nums': 1, 'names': ['present', 'blue areas', 'white areas', 'combinations'], 'abbrevs': 'PRS', 'scores': 1,
         'info': ''},
    ])