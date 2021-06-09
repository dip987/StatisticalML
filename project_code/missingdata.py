import numpy as np
import pandas as pd
import typing
from exploration_helper_functions import count_entries

"""
Column Names : 
Index(['Title', 'Genre', 'Tags', 'Languages', 'Series or Movie',
       'Country Availability', 'Runtime', 'Director', 'Writer', 'Actors',
       'Awards Received', 'Awards Nominated For', 'Release Date',
       'Netflix Release Date', 'Production House', 'Summary', 'IMDb Votes'],
"""


class MissingDataHandler:
    """
    Base Missing Data Handler Class. To extend this class, implement the process method
    """

    def __init__(self):
        self.replace_value = ''

    @staticmethod
    def process(data: pd.Series, fit_data: bool) -> None:
        raise NotImplementedError

    def reset(self):
        self.replace_value = ''


class ReplaceWithZero(MissingDataHandler):
    """
    Replace the missing data with 0.0
    """

    def __init__(self):
        super().__init__()
        self.replace_value = 0.0

    @staticmethod
    def process(data: pd.Series, fit_data=False):
        data.fillna(value=0.0, inplace=True)

    def reset(self):
        pass


class ReplaceWithHighestFrequency(MissingDataHandler):
    """
    Replaces with the highest occurring element. If multiple elements have the same highest count, keeps them all
    Only useful for string data
    """

    def __init__(self, keep_all: bool = True):
        super().__init__()
        self.keep_all = keep_all

    def process(self, data: pd.Series, fit_data=False):
        if fit_data:
            data_without_nan = data.dropna(axis=0)
            count_dict, _ = count_entries(data_without_nan)
            max_element = ''
            max_count = 0
            for element in count_dict.keys():
                if count_dict[element] > max_count:
                    max_count = count_dict[element]
                    max_element = element
                elif (count_dict[element] == max_count) & self.keep_all:
                    max_element += ', ' + element
            self.replace_value = max_element

        data.fillna(self.replace_value, inplace=True)

    def reset(self):
        self.replace_value = ''


class ReplaceWithValue(MissingDataHandler):
    """
    Replace All NaNs with the specified value
    """

    def __init__(self, replace_value):
        super().__init__()
        self.replace_value = replace_value

    def process(self, data: pd.Series, fit_data=False):
        data.fillna(value=self.replace_value, inplace=True)

    def reset(self):
        pass


class DoNothing(MissingDataHandler):
    @staticmethod
    def process(data: pd.Series, fit_data=False):
        pass


nan_string = 'unknown'
# By default all missing data are handled as zeros i.e. that category does not exist for that data


def handle_missing_data(df: pd.DataFrame, fit_data: bool, handling_scheme: typing.Dict = None) -> None:
    """
    Pass a dataframe and this function will modify the nan values according to the given scheme.(Inplace)

    :param df: The dataframe to operate on
    :param handling_scheme: A dictionary containing the column names and an individual MissingDataHandler object for
    that column. For missing column names, it uses the default scheme
    :param fit_data: Should the missing data handlers be updated using the data? Set to true for fit and fit_transform.
    Set to false for only transform
    :return: Modifies the given input dataframe
    """
    if handling_scheme is None:
        handling_scheme = {}
    # Drop na from the target variable
    df.dropna(axis=0, subset=['Title', 'IMDb Votes'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    for column_name in df.columns:
        assert column_name in handling_scheme, f"Unknown Column in the data named {column_name}"
        data_handler = handling_scheme[column_name]
        data_handler.process(df[column_name], fit_data=fit_data)


class DataImputer:
    def __init__(self, scheme: typing.Union[typing.Dict, None] = None):
        """
        Class to Impute data using custom missing data handlers.
        Pass a dictionary with (Column_name : MissingDataHandler()) and the imputer encoder will be used for the
        given columns. Otherwise default imputing classes will be used.(Default for each column is different)
        :param scheme: Dictionary mapping between (Column_name : MissingDataHandler()) class. The Encoding class
        must inherit from the MissingDataHandler class. There are many different instances of MissingDataHandler
        classes defined in this file. You can use them or make new classes that extend MissingDataHandler()
        """
        self.scheme = {'Genre': ReplaceWithValue(nan_string),
                       'Tags': ReplaceWithValue(nan_string),
                       'Languages': ReplaceWithValue(nan_string),
                       'Series or Movie': ReplaceWithValue(nan_string),
                       'Country Availability': ReplaceWithValue(nan_string),
                       'Runtime': ReplaceWithValue(nan_string),
                       'Director': ReplaceWithValue(nan_string),
                       'Writer': ReplaceWithValue(nan_string),
                       'Actors': ReplaceWithValue(nan_string),
                       'Awards Received': ReplaceWithZero(),
                       'Awards Nominated For': ReplaceWithZero(),
                       'Release Date': ReplaceWithHighestFrequency(keep_all=False),
                       'Netflix Release Date': ReplaceWithHighestFrequency(keep_all=False),
                       'Production House': ReplaceWithValue(nan_string),
                       'Title': DoNothing(),
                       'IMDb Votes': DoNothing()}
        if scheme is not None:
            self.scheme.update(scheme)
        self.has_been_fitted = False

    def fit_transform(self, dataframe):
        """
        Takes the entire dataframe and imputes it inplace using the scheme provided above. If any column name does not
        have a scheme, uses the default one. (The default schemes are different for each column). This also updates the
        parameters inside the MissingDataHandler classes

        :param dataframe: Pandas Dataframe containing both predictor and resultant variables
        """
        handle_missing_data(dataframe, fit_data=True, handling_scheme=self.scheme)
        self.has_been_fitted = True


    def transform(self, dataframe):
        """
        Transforms the given dataframe using pre-calculated missing data handlers. This does not change the
        pre-calculated parameters.

        """
        if self.has_been_fitted:
            handle_missing_data(dataframe, fit_data=False, handling_scheme=self.scheme)
        else:
            raise AssertionError("The Encoder Hasn't been fitted!")

    def fit(self, dataframe):
        """
        Takes the entire dataframe and imputes it using the scheme provided. Does not change the dataframe!
        If no scheme is provided for a specific column, the default scheme is used. The default is different for each
        column

        """
        temp_df = dataframe.copy()
        handle_missing_data(temp_df, fit_data=True, handling_scheme=self.scheme)
        self.has_been_fitted = True
