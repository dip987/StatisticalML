import typing
from scipy.sparse import csr_matrix
from path import Path
import numpy as np
import pandas as pd
import re
import os


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the Netflix data and drop out unnecessary columns
    :param path: Netflix Data Path
    :return: A Dataframe object
    """
    base_dir = Path(os.path.dirname(__file__)).parent
    filename = Path.joinpath(base_dir, path)
    df = pd.read_csv(filename)

    df.drop(columns=['Image', 'Poster', 'TMDb Trailer', 'Trailer Site', 'Netflix Link', 'IMDb Link', 'Hidden Gem Score',
                     'View Rating', 'IMDb Score', 'Rotten Tomatoes Score', 'Metacritic Score', 'Boxoffice', 'Summary'],
            inplace=True,
            errors='ignore')
    return df


def split_entries(series: pd.Series, mapping: typing.Union[None, typing.Mapping[str, int]] = None,
                  to_sparse: bool = True):
    """
    Takes the combined string entries of the series one by one, splits each row and encodes it into a sparse matrix
    :param series: The series to work on
    :param mapping: Pass a mapping from (category_name, int) to map the entries. If nothing is passed, the algorithm
    finds its own mapping and splits using that.(Passed mapping should start from 0)
    :param to_sparse: Set to false if you don't want a sparse representation
    :return: Tuple of The encoded matrix, mapping from each category to number, count for each category and
    total categories number
    """

    # If no mapping is passed, create the mapping. Grow the categories column as we go along
    if mapping is None:
        mapping = {}
        count = {}
        num_categories: int = 0
        encoded_matrix = []
        for index, entry in series.iteritems():
            if entry is not np.nan:
                split_entry = re.split(r",\s?", entry)
                for element in split_entry:
                    if element in mapping:
                        encoded_matrix[mapping[element]][index] = True
                        count[element] += 1
                    else:
                        mapping[element] = num_categories
                        num_categories += 1
                        count[element] = 1
                        encoded_matrix.append([False] * series.size)
                        encoded_matrix[mapping[element]][index] = True
        encoded_matrix = np.array(encoded_matrix)
        encoded_matrix = encoded_matrix.T
    else:
        count = {k: 0 for k in mapping.keys()}
        num_categories: int = max(list(mapping.values())) + 1
        encoded_matrix = np.zeros((series.shape[0], num_categories), dtype=bool)
        for index, entry in series.iteritems():
            if entry is not np.nan:
                split_entry = re.split(r",\s?", entry)
                for element in split_entry:
                    if element in mapping:
                        encoded_matrix[index, mapping[element]] = True
                        count[element] += 1
                    else:
                        pass

    if to_sparse:
        sparse_output = csr_matrix(encoded_matrix)
        return sparse_output, mapping, count, num_categories
    else:
        return encoded_matrix, mapping, count, num_categories


def count_entries(series: pd.Series) -> typing.Tuple[typing.Dict, int]:
    mapping = {}
    count = {}
    total_elements: int = 0
    for index, entry in series.items():
        split_entry = re.split(r",\s?", entry)
        for element in split_entry:
            if element in mapping:
                count[element] += 1
                pass
            else:
                mapping[element] = total_elements
                total_elements += 1
                count[element] = 1
    return count, total_elements


def split_entries_with_target_mean(data: pd.Series, target: np.ndarray,
                                   mapping: typing.Union[None, typing.Mapping[str, int]] = None,
                                   to_sparse: bool = True):
    """
    Takes the combined string entries of the series one by one, splits each row and encodes it into a sparse matrix.
    Also calculate the mean of the target value for each individual category!
    :param data: The series to work on
    :param target: The target/resultant variable
    :param mapping: Pass a mapping from (category_name, int) to map the entries. If nothing is passed, the algorithm
    finds its own mapping and splits using that.(Passed mapping should start from 0)
    :param to_sparse: Set to false if you don't want a sparse representation
    :return: Tuple of The encoded matrix, mapping from each category to number, count for each category, the mean of
    the target variable and total categories number
    """
    if mapping is None:
        mapping = {}
        count = {}
        target_mean = {}
        num_categories: int = 0
        encoded_matrix = []
        for index, entry in data.iteritems():
            if entry is not np.nan:
                split_entry = re.split(r",\s?", entry)
                for element in split_entry:
                    if element in mapping:
                        encoded_matrix[mapping[element]][index] = True
                        count[element] += 1
                        # The target_predict keeps the average target value for that specific category
                        target_mean[element] += (target[index] - target_mean[element]) / count[element]
                    else:
                        mapping[element] = num_categories
                        num_categories += 1
                        count[element] = 1
                        target_mean[element] = target[index]
                        encoded_matrix.append([False] * data.size)
                        encoded_matrix[mapping[element]][index] = True
        encoded_matrix = np.array(encoded_matrix)
        encoded_matrix = encoded_matrix.T
    else:
        target_mean = {k: 0 for k in mapping.keys()}
        count = {k: 0 for k in mapping.keys()}
        num_categories: int = max(list(mapping.values())) + 1
        encoded_matrix = np.zeros((data.shape[0], num_categories), dtype=bool)
        for index, entry in data.iteritems():
            if entry is not np.nan:
                split_entry = re.split(r",\s?", entry)
                for element in split_entry:
                    if element in mapping:
                        encoded_matrix[index, mapping[element]] = True
                        count[element] += 1
                        target_mean[element] += (target[index] - target_mean[element]) / count[element]
                    else:
                        pass

    if to_sparse:
        sparse_output = csr_matrix(encoded_matrix)
        return sparse_output, mapping, count, target_mean, num_categories
    else:
        return encoded_matrix, mapping, count, target_mean, num_categories


def custom_tokenizer(text):
    split_text = re.split(r",\s?", text)
    return split_text
