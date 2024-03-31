from pandas import DataFrame
from PIL import Image
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Derm7pt_data(Dataset):
    #multiple specific class names grouped together by more general names
    diagnosis: DataFrame = pd.DataFrame([
        {'nums': 0, 'is_cancer': 1, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'nums': 1, 'is_cancer': 0,
         'names': ['nevus', 'blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus',
                   'recurrent nevus', 'reed or spitz nevus'], 'abbrevs': 'NEV', 'info': 'Moles'},
        {'nums': 2, 'is_cancer': 1,
         'names': ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                   'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)', 'melanoma metastasis'],
         'abbrevs': 'MEL', 'info': 'Melanoma, cancer'},
        {'nums': 3, 'is_cancer': 0, 'names': ['DF/LT/MLS/MISC', 'dermatofibroma', 'lentigo', 'melanosis'
            , 'miscellaneous', 'vascular lesion'],
         'abbrevs': 'MISC', 'info': 'benign'},
        {'nums': 4, 'is_cancer': 0, 'names': 'seborrheic keratosis', 'abbrevs': 'SK', 'info': 'benign'},
    ])

    #Columns used in learning process for model
    #Potential concept columns ['location', 'sex']
    model_columns = {
        'concepts': ['pigment_network', 'streaks', 'pigmentation', 'regression_structures', 'dots_and_globules',
                     'blue_whitish_veil', 'vascular_structures'],
        'label': "nums",
    }

    def __init__(self, data_folder: str):
        self.image_size = (768, 512)
        self.image_folder = os.path.join(os.path.normpath(data_folder), "images")
        self.metadata = self.loadMeta(os.path.join(data_folder, "meta\\meta.csv"))
        if self.metadata.empty:
            self.labels = pd.DataFrame(columns=self.model_columns["label"])
        else:
            self.labels = self.metadata[self.model_columns["label"]]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, index):
        #Todo Metadata/Concepts
        metadata = self.metadata.iloc[index]

        data = self.loadImage(metadata['derm'])
        # Convert the Pandas Series to a NumPy array and then to a PyTorch tensor
        data = self.transform(data)

        label = metadata[self.model_columns["label"]]

        return data, label

    def loadImage(self, file: str):
        """
        Load image and resize it to 768x512

        :param file:
        :return: PixelImage
        """
        file = os.path.normpath(file)
        img = Image.open(os.path.join(self.image_folder, file))
        img = img.resize(self.image_size)
        return img

    def loadImages_ToDo(self):
        #ToDo: add both clinical and dermoscopic images into dataset
        pass


    #load meta data
    def loadMeta(self, file_path: str):
        merged_df = pd.DataFrame()
        if os.path.exists(file_path):
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

#ToDo
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