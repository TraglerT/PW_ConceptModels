from pandas import DataFrame
from PIL import Image
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Derm7pt_data(Dataset):
    #Columns used in learning process for model
    #Potential concept columns ['location', 'sex']
    model_columns = {
        'concepts': ['pigment_network_num', 'streaks_num', 'pigmentation_num', 'regression_structures_num',
                     'dots_and_globules_num', 'blue_whitish_veil_num', 'vascular_structures_num'],
        'label': "diagnosis_num",
    }

    #multiple specific class names grouped together by more general names
    diagnosis_mapping: DataFrame = pd.DataFrame([
        {'diagnosis_num': 0, 'is_cancer': 1, 'names': 'basal cell carcinoma', 'abbrevs': 'BCC', 'info': 'Common non-melanoma cancer'},
        {'diagnosis_num': 1, 'is_cancer': 0,
         'names': ['nevus', 'blue nevus', 'clark nevus', 'combined nevus', 'congenital nevus', 'dermal nevus',
                   'recurrent nevus', 'reed or spitz nevus'], 'abbrevs': 'NEV', 'info': 'Moles'},
        {'diagnosis_num': 2, 'is_cancer': 1,
         'names': ['melanoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                   'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)', 'melanoma metastasis'],
         'abbrevs': 'MEL', 'info': 'Melanoma, cancer'},
        {'diagnosis_num': 3, 'is_cancer': 0, 'names': ['DF/LT/MLS/MISC', 'dermatofibroma', 'lentigo', 'melanosis'
            , 'miscellaneous', 'vascular lesion'],
         'abbrevs': 'MISC', 'info': 'benign'},
        {'diagnosis_num': 4, 'is_cancer': 0, 'names': 'seborrheic keratosis', 'abbrevs': 'SK', 'info': 'benign'},
    ])

    #Concepts (used grouped concepts for pigmentation, regression structures, and vascular structures)
    concepts_mapping = {
        'pigment_network': pd.DataFrame([
            {'pigment_network_num': 0, 'names': ['absent', 'typical'], 'pigment_network_score': 0},
            {'pigment_network_num': 1, 'names': 'atypical', 'pigment_network_score': 2},
        ]),
        'streaks': pd.DataFrame([
            {'streaks_num': 0, 'names': ['absent', 'regular'], 'streaks_score': 0},
            {'streaks_num': 1, 'names': 'irregular', 'streaks_score': 1},
        ]),
        'pigmentation': pd.DataFrame([
            {'pigmentation_num': 0, 'names': ['absent', 'regular', 'diffuse regular', 'localized regular'], 'pigmentation_score': 0},
            {'pigmentation_num': 1, 'names': ['irregular', 'diffuse irregular', 'localized irregular'], 'pigmentation_score': 1},
        ]),
        'regression_structures': pd.DataFrame([
            {'regression_structures_num': 0, 'names': 'absent', 'regression_structures_score': 0},
            {'regression_structures_num': 1, 'names': ['present', 'blue areas', 'white areas', 'combinations'],
                'regression_structures_score': 1},
        ]),
        'dots_and_globules': pd.DataFrame([
            {'dots_and_globules_num': 0, 'names': ['absent', 'regular'], 'dots_and_globules_score': 0},
            {'dots_and_globules_num': 1, 'names': 'irregular', 'dots_and_globules_score': 1},
        ]),
        'blue_whitish_veil': pd.DataFrame([
            {'blue_whitish_veil_num': 0, 'names': 'absent', 'blue_whitish_veil_score': 0},
            {'blue_whitish_veil_num': 1, 'names': 'present', 'blue_whitish_veil_score': 2},
        ]),
        'vascular_structures': pd.DataFrame([
            {'vascular_structures_num': 0, 'names': ['regular', 'arborizing', 'comma', 'hairpin',
                                                     'within regression', 'wreath', 'absent'], 'vascular_structures_score': 0},
            {'vascular_structures_num': 1, 'names': ['dotted/irregular', 'dotted', 'linear irregular'], 'vascular_structures_score': 2},
        ]),
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
        metadata = self.metadata.iloc[index]

        data = self.loadImage(metadata['derm'])
        # Convert the Pandas Series to a NumPy array and then to a PyTorch tensor
        data = self.transform(data)

        label = metadata[self.model_columns["label"]]
        concepts = pd.to_numeric(metadata[self.model_columns["concepts"]], errors='ignore')
        concepts = torch.tensor(concepts.values, dtype=torch.float32)

        return data, label, concepts

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


    #load meta data
    def loadMeta(self, file_path: str):
        merged_df = pd.DataFrame()
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            #drop columns that are not needed
            drop_columns = ['case_id', 'notes', 'management']
            df = df.drop(drop_columns, axis=1)
            #diagnosis is very specific, so we merge it with the more general diagnosis grouping.
            merged_df = df.merge(self.diagnosis_mapping.explode('names'), how='left', left_on='diagnosis', right_on='names')
            merged_df = merged_df.drop('names', axis=1)

            #concepts Merger
            for concept_name, mapping_df in self.concepts_mapping.items():
                merged_df = merged_df.merge(mapping_df.explode('names'), how='left', left_on=concept_name, right_on='names')
                merged_df = merged_df.drop('names', axis=1)
        return merged_df





#   multi_modal_concepts_mapping = {
#         'pigment_network': pd.DataFrame([
#             {'pigment_network_num': 0, 'names': 'absent', 'pigment_network_score': 0},
#             {'pigment_network_num': 1, 'names': 'typical', 'pigment_network_score': 0},
#             {'pigment_network_num': 2, 'names': 'atypical', 'pigment_network_score': 2},
#         ]),
#         'streaks': pd.DataFrame([
#             {'streaks_num': 0, 'names': 'absent', 'streaks_score': 0},
#             {'streaks_num': 1, 'names': 'regular', 'streaks_score': 0},
#             {'streaks_num': 2, 'names': 'irregular', 'streaks_score': 1},
#         ]),
#         'pigmentation': pd.DataFrame([
#             {'pigmentation_num': 0, 'names': 'absent', 'pigmentation_score': 0},
#             {'pigmentation_num': 1, 'names': ['regular', 'diffuse regular', 'localized regular'], 'pigmentation_score': 0},
#             {'pigmentation_num': 2, 'names': ['irregular', 'diffuse irregular', 'localized irregular'], 'pigmentation_score': 1},
#         ]),
#         'regression_structures': pd.DataFrame([
#             {'regression_structures_num': 0, 'names': 'absent', 'regression_structures_score': 0},
#             {'regression_structures_num': 1, 'names': ['present', 'blue areas', 'white areas', 'combinations'],
#                 'regression_structures_score': 1},
#         ]),
#         'dots_and_globules': pd.DataFrame([
#             {'dots_and_globules_num': 0, 'names': 'absent', 'dots_and_globules_score': 0},
#             {'dots_and_globules_num': 1, 'names': 'regular', 'dots_and_globules_score': 0},
#             {'dots_and_globules_num': 2, 'names': 'irregular', 'dots_and_globules_score': 1},
#         ]),
#         'blue_whitish_veil': pd.DataFrame([
#             {'blue_whitish_veil_num': 0, 'names': 'absent', 'blue_whitish_veil_score': 0},
#             {'blue_whitish_veil_num': 1, 'names': 'present', 'blue_whitish_veil_score': 2},
#         ]),
#         'vascular_structures': pd.DataFrame([
#             {'vascular_structures_num': 0, 'names': 'absent', 'vascular_structures_score': 0},
#             {'vascular_structures_num': 1, 'names': ['regular', 'arborizing', 'comma', 'hairpin',
#                                                      'within regression', 'wreath'], 'vascular_structures_score': 0},
#             {'vascular_structures_num': 2, 'names': ['dotted/irregular', 'dotted', 'linear irregular'], 'vascular_structures_score': 2},
#         ]),
#     }