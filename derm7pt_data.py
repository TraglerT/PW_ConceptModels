from pandas import DataFrame
from PIL import Image
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Derm7pt_data(Dataset):
    """
    Custom dataset class for loading and transforming data for the Derm7pt model. This class loads image and metadata
    information from a specified folder, applies data preprocessing, and groups specific diagnoses and dermatoscopic
    features(concepts) together.

    Attributes:
        model_columns (dict): Dictionary specifying the columns for 'concepts' and 'label' used in the model.
        diagnosis_mapping (DataFrame): Mapping table that categorizes specific diagnoses (labels) into more general diagnostic categories.
        concepts_mapping (dict): Dictionary that maps different sub-concepts into general concepts and assigns them <name>_num = 1 if the concept is present.

    Sub-Attributes:
        <name>_num (Boolean): Indicates if concept or label present.
        is_cancer (Boolean): Is label cancerous or not.
        names (list of str): Corresponds to the values within the csv file that represent the type of label. Multiple names are grouped together.
        abbrevs (str): a unique abbreviation that represents the label.
        info (str): Helpful information. Not otherwise used.
        <name>_score (int): 7-point criteria score given to that concept. Not used in the implementation.
    """

    # Column names for concepts and labels, used to extract data from metadata
    model_columns = {
        'concepts': ['pigment_network_num', 'streaks_num', 'pigmentation_num', 'regression_structures_num',
                     'dots_and_globules_num', 'blue_whitish_veil_num', 'vascular_structures_num'],
        'label': "diagnosis_num",
    }

    ### cite: used code from Kawahara derm7pt Github as basis for this part (https://github.com/jeremykawahara/derm7pt) ###
    # multiple specific class names grouped together by more general names
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

    # Concepts (used grouped concepts for pigmentation, regression structures, and vascular structures)
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

    ### end cite ###

    def __init__(self, data_folder: str):
        """
        Initialize the dataset with image and metadata paths and load metadata information from the given path.
        Parameters:
            data_folder (str): Directory containing images and metadata files.
        """
        self.image_size = (768, 512)
        self.image_folder = os.path.join(os.path.normpath(data_folder), "images")
        self.metadata = self._loadMeta(os.path.join(data_folder, "meta//meta.csv"))
        if self.metadata.empty:
            self.labels = pd.DataFrame(columns=self.model_columns["label"])
        else:
            self.labels = self.metadata[self.model_columns["label"]]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        """
        Return the number of records in the metadata (number of samples in the dataset).

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve the image, label, and concepts for a specific index in the dataset.

        Parameters:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing (image data as tensor, label, concepts as tensor).
        """
        metadata = self.metadata.iloc[index]

        data = self.loadImage(metadata['derm'])
        # Convert the Pandas Series to a NumPy array and then to a PyTorch tensor
        data = self.transform(data)

        # Retrieve label and concept values for the sample
        label = metadata[self.model_columns["label"]]
        concepts = pd.to_numeric(metadata[self.model_columns["concepts"]])
        concepts = torch.tensor(concepts.values, dtype=torch.float32)

        return data, label, concepts

    def loadImage(self, file: str):
        """
        Load image from file and resize it to self.image_size(768x512)

        Parameters:
            file (str): Filename of the image.

        Returns:
            PIL.Image.Image: Loaded and resized image.
        """
        file = os.path.normpath(file)
        path = os.path.join(self.image_folder, file)
        img = Image.open(self.__find_case_insensitive_path(path))
        img = img.resize(self.image_size)
        return img

    def _loadMeta(self, file_path: str) -> pd.DataFrame:
        """
        Load metadata from a CSV file, drop unnecessary columns, and merge specific diagnosis and concept mappings.

        Parameters:
            file_path (str): Path to the metadata CSV file.

        Returns:
            pd.DataFrame: DataFrame containing processed metadata.
        """
        merged_df = pd.DataFrame()
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # drop columns that are not needed
            drop_columns = ['case_id', 'notes', 'management']
            df = df.drop(drop_columns, axis=1)

            # Merge specific diagnoses with the general diagnosis mapping
            merged_df = df.merge(self.diagnosis_mapping.explode('names'), how='left', left_on='diagnosis',
                                 right_on='names')
            merged_df = merged_df.drop('names', axis=1)

            # Merge concept mappings
            for concept_name, mapping_df in self.concepts_mapping.items():
                merged_df = merged_df.merge(mapping_df.explode('names'), how='left', left_on=concept_name,
                                            right_on='names')
                merged_df = merged_df.drop('names', axis=1)
        return merged_df

    def __find_case_insensitive_path(self, path: str) -> str:
        """
        Find the correct case-sensitive path for systems with case-sensitive file systems.

        Parameters:
            path (str): Path to the file with case-insensitive naming.

        Returns:
            str: Correct case-sensitive file path if found.
        """
        if os.path.isfile(path):
            return path

        # Check case-sensitivity in the last two path elements
        parts = path.split(os.sep)
        current_path = os.path.join(os.sep, os.path.join(*parts[1:-2]))
        for part in parts[-2:]:
            entries = os.listdir(current_path)
            for entry in entries:
                if entry.lower() == part.lower():
                    current_path = os.path.join(current_path, entry)
                    break
        return current_path