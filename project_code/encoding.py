import numpy
import pandas as pd
import numpy as np
import typing

import scipy.sparse
from scipy.sparse import csr_matrix
from exploration_helper_functions import split_entries, split_entries_with_target_mean
from datetime import datetime
import re


class BaseEncoding:
    """
    Base Class for encoding features into a machine-readable algorithm-friendly format
    """

    def __init__(self):
        self.category_names = []

    def encode(self, data: pd.Series, target: np.ndarray, update: bool) -> scipy.sparse.csr_matrix:
        """
        Encode the given series and return a Sparse matrix

        :param target: The target y
        :param data: Pandas Series data
        :param update: Should the saved encoding parameters be updated? This includes mapping, category names and so on.
        Set to true when doing fit or fit_transform. When doing just transform, set the fit_data to false.
        :return: Sparse matrix
        """
        raise NotImplementedError

    def get_category_mapping(self) -> typing.Mapping:
        """
        Get a Category to integer mapping for this encoder once encoding is done. It can then be used for
        mapping/reverse encoding etc.. If this encoding class does not contain any categories, returns an empty mapping

        :return: Mapping between category name to index
        """
        if self.category_names:
            return {category: index for index, category in enumerate(self.category_names)}
        else:
            return {}


class DoNothing(BaseEncoding):
    """
    Encoding returns the same output as the input series(But converted into a sparse matrix)
    """

    def encode(self, data: pd.Series, target: np.ndarray = None, update=False):
        return scipy.sparse.csr_matrix(data.to_numpy(copy=True).reshape(-1, 1))


class DropColumn(BaseEncoding):
    """
    Drops this column from the feature matrix
    """

    def encode(self, data: pd.Series, target: np.ndarray = None, update=False):
        return scipy.sparse.csr_matrix(numpy.empty(shape=(data.shape[0], 0)))


class OneHot(BaseEncoding):
    """
    Classic One Hot Encoding. For this project, there might be multiple categories per entry. Breaks them up and puts a
    1.0 for each column that the algorithm finds in for each entry
    """

    def encode(self, data: pd.Series, target: np.ndarray = None, update=True):
        if update:
            encoded_matrix, mapping, _, __ = split_entries(data)
            self.category_names = list(mapping.keys())
        else:
            encoded_matrix, _, __, ___ = split_entries(data, mapping=self.get_category_mapping())
        return encoded_matrix


class KeepTopN(BaseEncoding):
    """
    Keeps only the Top n category encodings and kills the rest. i.e. if a row does not contain any one of these
    categories it will have all zeroes for this feature
    """

    def __init__(self, N: int = 5):
        super().__init__()
        self.N = N

    def encode(self, data: pd.Series, target: np.ndarray = None, update=True):
        if update:
            encoded_matrix, mapping, count, _ = split_entries(data, to_sparse=False)
            # Make sure n is small enough. If it's too big, set it to the largest possible value
            self.N = min(self.N, len(mapping))
            # Sort according to count and keep the largest n categories
            count = dict(sorted(count.items(), key=lambda item: item[1]))
            top_N_categories = list(count.keys())[-self.N:]
            self.category_names = top_N_categories
            top_N_indices = [mapping[category] for category in top_N_categories]
            encoded_matrix = encoded_matrix[:, top_N_indices]
            encoded_matrix = scipy.sparse.csr_matrix(encoded_matrix)
        else:
            encoded_matrix, _, __, ___ = split_entries(data, mapping=self.get_category_mapping())
        return encoded_matrix


class DateEncoding(BaseEncoding):
    def __init__(self, format_string: str = r"%d %b %Y", year_offset: int = 2000, month_encoding: str = 'ordinal'):
        """
        Encodes the day, month and year separately. The day and month gets converted to (0, 1] and the year gets an
        offset from year_offset(1900) and then divided by a normalizing factor. This makes the offset year go to 0.0
        and the current year(2021) to become 1.0 after encoding

        :param format_string: How to read the date from the entry. This is different for the two separate datetime
        columns we have in the dataset
        :param year_offset: encoded year = (year - year_offset)/(normalizing factor) => [0, 1.0]
        :param month_encoding: 'ordinal' or 'one_hot' , how to encode the month
        """
        super().__init__()
        self.format_string = format_string
        self.year_offset = year_offset
        self.category_names = ['Day', 'Month', 'Offset Year']
        self.month_max_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        self.month_encoding = month_encoding
        self.normalizing_factor = 2021 - year_offset

    def encode(self, data: pd.Series, target: np.ndarray = None, update=False):
        if self.month_encoding == 'ordinal':
            encoded_matrix = np.zeros((data.size, 3))
            temp_data = data.map(lambda string: datetime.strptime(string, self.format_string))
            for index, date in temp_data.items():
                encoded_matrix[index, 0] = date.day / self.month_max_days[date.month]
                encoded_matrix[index, 1] = date.month / 12
                encoded_matrix[index, 2] = (date.year - self.year_offset) / self.normalizing_factor
        elif self.month_encoding == 'one_hot':
            encoded_matrix = np.zeros((data.size, 14))
            temp_data = data.map(lambda string: datetime.strptime(string, self.format_string))
            for index, date in temp_data.items():
                encoded_matrix[index, 0] = date.day / self.month_max_days[date.month]
                encoded_matrix[index, date.month] = 1.0
                encoded_matrix[index, -1] = (date.year - self.year_offset) / self.normalizing_factor
        else:
            raise AssertionError("Unknown Month Encoding Scheme!")

        return scipy.sparse.csr_matrix(encoded_matrix)


class NormalizedNumberEncoding(BaseEncoding):
    """
    Used for encoding raw numbers. Normalizes the numbers to be between 0.0 and 1.0
    """

    def __init__(self):
        super().__init__()
        self.data_max = 10
        self.data_min = 0

    def encode(self, data: pd.Series, target: np.ndarray = None, update=True):
        if update:
            self.data_max = data.max()
            self.data_min = data.min()
        normalized_data = (data.to_numpy(copy=True).reshape(-1, 1) - self.data_min) / (self.data_max - self.data_min)
        return scipy.sparse.csr_matrix(normalized_data)


class TargetPriorityNEncoding(BaseEncoding):
    """
    Correlates the average value of the target for each category in the feature. This gives a measure of the impact on
    the target due to that specific category. Keeps the top n most impactful categories and drops the rest
    (This takes a while to encode)
    """

    def __init__(self, n: typing.Union[str, int]):
        super(TargetPriorityNEncoding, self).__init__()
        self.target_means = []
        self.N = n

    def encode(self, data: pd.Series, target: np.ndarray = None, update: bool = True) -> scipy.sparse.csr_matrix:

        # Make sure when we want to update, there is a target passed into the function
        if (target is None) and (update is True):
            raise AssertionError("Need to provide a target variable if you're training")
        if update is True:
            encoded_matrix, mapping, count, target_mean, num_categories = split_entries_with_target_mean(
                data=data, target=target, to_sparse=False)
            # Make sure n is small enough. If it's too big, set it to the largest possible value
            self.N = min(self.N, len(mapping))
            # Sort according to count and keep the largest n categories
            target_mean = dict(sorted(target_mean.items(), key=lambda item: item[1]))
            top_N_categories = list(target_mean.keys())[-self.N:]
            self.category_names = top_N_categories
            self.target_means = [target_mean[category] for category in self.category_names]
            top_N_indices = [mapping[category] for category in top_N_categories]
            encoded_matrix = encoded_matrix[:, top_N_indices]
            encoded_matrix = scipy.sparse.csr_matrix(encoded_matrix)
            return encoded_matrix
        else:
            encoded_matrix, _, __, ___, ____ = split_entries_with_target_mean(data=data,
                                                                              target=target,
                                                                              mapping=self.get_category_mapping(),
                                                                              to_sparse=True)
            return encoded_matrix


def handle_encoding(df: pd.DataFrame, fit_data: bool, handling_scheme=None,
                    has_y: bool = True) -> typing.Tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """
    Gets the feature matrix X and the resultant variable Y array from the dataframe

    :param df: The dataframe to operate on
    :param handling_scheme: A dictionary containing the column names and BaseEncoding object for that column. For any
    missing columns, the default encoding scheme is used
    :param fit_data: Update the encoding according to the data? If false, the encoding classes' params. do not change.
    Set this to true only when fit or fit_transform. If we are just doing a transform, set to false
    :param has_y: Does the input dataframe include the resultant variable y?
    :return: A new numpy array X and Y
    """
    if handling_scheme is None:
        handling_scheme = {}
    X = scipy.sparse.csr_matrix(np.empty(shape=(df.shape[0], 0)))

    # Handle the result variable
    if has_y:
        y = df['IMDb Votes'].to_numpy(copy=True) / (df['IMDb Votes'].max())
    else:
        y = np.zeros((df.shape[0], 1))

    # Process each column one by one by using the BaseEncoding objects inside the handling_scheme
    for column_name in df.columns:
        # Ignore these 2 columns in data processing
        if column_name == 'Title':
            continue
        if column_name == 'IMDb Votes':
            continue

        # Ensure the handling_scheme actually has the specified column
        assert column_name in handling_scheme, f"The Scheme does not have a way to handle {column_name}"

        # Get the Column Encoder
        data_encoder = handling_scheme[column_name]
        encoded_data = data_encoder.encode(df[column_name], target=y, update=fit_data)
        X = scipy.sparse.hstack([X, encoded_data])
    return X, y


class DataEncoder:
    def __init__(self, scheme=None):
        """
        Class to Encode data using custom encoders. Pass a dictionary with (Column_name : BaseEncoding()) and the
        corresponding encoder will be used for the given columns. Otherwise default encoding classes will be used.

        :param scheme: Dictionary mapping between (Column_name : Encoding()) class. The Encoding class must inherit from
        the BaseEncoding class. There are many different instances of BaseEncoding class defined in this file. You can
        use them or make new classes that extend BaseEncoding()
        """
        self.scheme = {'Genre': KeepTopN(N=20),
                       'Tags': KeepTopN(N=20),
                       'Languages': KeepTopN(N=20),
                       'Series or Movie': KeepTopN(N=20),
                       'Country Availability': KeepTopN(N=20),
                       'Runtime': KeepTopN(N=20),
                       'Director': KeepTopN(N=20),
                       'Writer': KeepTopN(N=20),
                       'Actors': KeepTopN(N=20),
                       'Awards Received': NormalizedNumberEncoding(),
                       'Awards Nominated For': NormalizedNumberEncoding(),
                       'Release Date': DateEncoding(format_string=r"%d %b %Y"),
                       'Netflix Release Date': DateEncoding(format_string=r"%Y-%m-%d"),
                       'Production House': KeepTopN(N=20),
                       }
        if scheme is not None:
            self.scheme.update(scheme)
        self.has_been_fitted = False

    def fit_transform(self, dataframe):
        """
        Takes the entire dataframe after imputation and encodes it using the scheme specified before. Also fits the
        encoding parameters using this data. This includes category names, standardization, etc.
        Returns both the predictor and resultant variables.

        :param dataframe: Pandas Dataframe containing both predictor and resultant variables
        :return: X, y
        """
        x, y = handle_encoding(dataframe, fit_data=True, handling_scheme=self.scheme, has_y=True)
        self.has_been_fitted = True
        return x, y

    def transform(self, dataframe):
        """
        Transforms the given dataframe using pre-calculated encodings. This does not change the pre-calculated encoing
        parameters.

        :return: X, y
        """
        if self.has_been_fitted:
            x, y = handle_encoding(dataframe, fit_data=False, handling_scheme=self.scheme, has_y=True)
            return x, y
        else:
            raise AssertionError("The Encoder Hasn't been fitted!")

    def fit(self, dataframe):
        """
        Takes the entire dataframe after imputation and fits the encoding parameters using the given data

        :param dataframe: Pandas Dataframe containing both predictor and resultant variables
        """
        x, y = handle_encoding(dataframe, fit_data=True, handling_scheme=self.scheme, has_y=True)
        self.has_been_fitted = True
