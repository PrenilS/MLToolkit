# Create a python class to impliment the automated machine learning pipeline
# It should take a dataframe as input, what the target variable is as a string, a bit input to say whether data cleaning should be done, a bit for whether to use a specified training, test and validation split or if the training, test and validation should be split by date and which cutoffs to use. It should also take a list with options per feature for 'normalise', 'onehot', 'woe' and 'bin' to say that each feature should be normalised, one-hot encoded, replaced with weight of evidence values or binned into ordinal categories.
# It should then try to perform different classification models, regression models and time series models to determine the best. For classification, it should try random under and over sampling to equalise classes. It should then output a final best model object but also make it so that the history of all attempted models and their relevant performance metrics can be retrieved via a call
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
# Add other models as needed

class AutomatedMachineLearning:
    def __init__(self, target_variable):
        self.target_variable = target_variable
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_model = None
        self.model_history = {}

    # load data from a dataframe
    def load_data(self, dataframe):
        self.data = dataframe

    # clean data: Look for missing values, outliers, duplicates, etc. and implement logic to deal with them
    def data_cleaning(self, null_feat_method_num:str='drop', null_feat_method_cat:str='drop', null_target_method:str='drop', missing_thresh:float=0.7,
                      num_const:float=0.0, cat_const:str='Missing', outlier_method:str='drop', outlier_threshold:float=3.0):
        """
        Perform data cleaning on the loaded DataFrame.

        This function applies various data cleaning operations to the loaded DataFrame to prepare it
        for further processing and modeling.

        Parameters:
            dataframe (pd.DataFrame): DataFrame to be used for modeling.
            null_feat_method_num (str): Method to deal with null values in features. Options are 'drop', 'mean', 'median', 'mode', 'constant'
            null_feat_method_cat (str): Method to deal with null values in features. Options are 'drop', 'mode', 'constant'
            null_target_method (str): Method to deal with null values in target
            num_const (float): Constant value to replace null values in numerical features
            missing_thresh (float): Threshold for dropping features with too many missing values
            cat_const (str): Constant value to replace null values in categorical features
            outlier_method (str): Method to deal with outliers in numerical features. Options are 'drop', 'mean', 'median', 'mode' and 'none'
            outlier_threshold (float): Threshold for dropping outliers in numerical features


        Returns:
            None

        Examples:
            # Assuming 'automl' is an instance of AutomatedMachineLearning class
            automl.data_cleaning()
        """
        # Drop features with too many missing values
        self.data.dropna(axis=1, thresh=missing_thresh*len(self.data), inplace=True)
        # Drop rows with missing values in target
        self.data.dropna(subset=[self.target_variable], inplace=True)
         # Separate numerical and categorical features
        numerical_features = self.data.select_dtypes(include='number')
        categorical_features = self.data.select_dtypes(exclude='number')\
        
        # save list of numerical and categorical features to self
        self.numerical_features = list(numerical_features.columns)
        self.categorical_features = list(categorical_features.columns)

        # Handle numerical features
        if null_feat_method_num == 'drop':
            numerical_features.dropna(inplace=True)
        elif null_feat_method_num == 'mean':
            numerical_features.fillna(numerical_features.mean(), inplace=True)
        elif null_feat_method_num == 'median':
            numerical_features.fillna(numerical_features.median(), inplace=True)
        elif null_feat_method_num == 'constant':
            numerical_features.fillna(num_const, inplace=True)  # Replace '0' with your desired constant value
        else:
            raise ValueError('Invalid value for null_feat_method. Valid values are \'drop\', \'mean\', \'median\', and \'constant\'')

        # Handle categorical features
        if null_feat_method_cat == 'drop':
            categorical_features.dropna(inplace=True)
        elif null_feat_method_cat == 'mode':
            categorical_features.fillna(categorical_features.mode().iloc[0], inplace=True)  # Impute with the mode
        elif null_feat_method_cat == 'constant':
            categorical_features.fillna(cat_const, inplace=True)  # Replace 'constant_value' with your desired constant value
        else:
            raise ValueError('Invalid value for null_feat_method. Valid values are \'drop\', \'mode\', and \'constant\'')

        # Combine the modified numerical and categorical features back into the main DataFrame
        self.data = pd.concat([numerical_features, categorical_features], axis=1)

        # Remove outliers from numerical features
        if outlier_method == 'drop':
            for feature in self.numerical_features:
                self.data = self.data[(self.data[feature] - self.data[feature].mean()) / self.data[feature].std() < outlier_threshold]
        elif outlier_method == 'mean':
            for feature in self.numerical_features:
                self.data.loc[(self.data[feature] - self.data[feature].mean()) / self.data[feature].std() >= outlier_threshold, feature] = self.data[feature].mean()
        elif outlier_method == 'mode':
            for feature in self.numerical_features:
                self.data.loc[(self.data[feature] - self.data[feature].mean()) / self.data[feature].std() >= outlier_threshold, feature] = self.data[feature].mode()
        elif outlier_method == 'median':
            for feature in self.numerical_features:
                self.data.loc[(self.data[feature] - self.data[feature].mean()) / self.data[feature].std() >= outlier_threshold, feature] = self.data[feature].median()
        # Remove duplicates
        self.data.drop_duplicates(inplace=True)




    def split_data(self, date_col: str, date_split: bool = False, train_cutoff_date: str = None, test_cutoff_date: str = None, val_cutoff_date: str = None,
                    train_cutoff_prop: float = None, test_cutoff_prop: float = None):
        """
        Split dataset into training, test and valuation sets.

        This function splits the dataset into training, test, and valuation sets based on the options provided.

        Parameters:
            date_col (str): Name of the column containing dates
            date_split (bool): Whether to split the dataset based on dates.
            train_cutoff_date (str): Date to split training set on
            test_cutoff_date (str): Date to split test set on
            val_cutoff_date (str): Date to split valuation set on
            train_cutoff_prop (float): Proportion of data to use for training
            test_cutoff_prop (float): Proportion of data to use for test

        Returns:
            None

        Examples:
            # Assuming 'automl' is an instance of AutomatedMachineLearning class
            automl.split_data()
        """
        self.X = self.data.drop(self.target_variable, axis=1)
        self.y = self.data[self.target_variable]

        if date_split:
            # Split the data based on dates
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data = self.data.sort_values(by=date_col)
            self.X_train = self.data[self.data[date_col] < pd.to_datetime(train_cutoff_date)]
            self.X_test = self.data[(self.data[date_col] >= pd.to_datetime(train_cutoff_date)) & (self.data[date_col] < pd.to_datetime(test_cutoff_date))]
            self.X_val = self.data[self.data[date_col] >= pd.to_datetime(test_cutoff_date)]
            self.y_train = self.y[self.data[date_col] < pd.to_datetime(train_cutoff_date)]
            self.y_test = self.y[(self.data[date_col] >= pd.to_datetime(train_cutoff_date)) & (self.data[date_col] < pd.to_datetime(test_cutoff_date))]
            self.y_val = self.y[self.data[date_col] >= pd.to_datetime(test_cutoff_date)]
        else:
            # Split the data based on proportion
            val_cutoff_prop = 1.0 - train_cutoff_prop - test_cutoff_prop
            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(self.X, self.y, train_size=train_cutoff_prop, test_size=(1.0 - train_cutoff_prop))
            self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_temp, self.y_temp, train_size=test_cutoff_prop / (test_cutoff_prop + val_cutoff_prop), test_size=val_cutoff_prop / (test_cutoff_prop + val_cutoff_prop))

    def feature_engineering(self, feature_options_num:str = 'norm', feature_options_cat:str='onehot', num_bins:int=10):
        """
        Preprocess features based on the options provided.

        Parameters:
            feature_options_num (list): List of preprocessing options for each feature. Options are 'norm', 'onehot', 'woe' and 'bin'
            feature_options_cat (list): List of preprocessing options for each feature. Options are 'norm', 'onehot', 'woe' and 'bin'

        Returns:
            None

        Examples:
            # Assuming 'automl' is an instance of AutomatedMachineLearning class
            automl.feature_engineering()
        """
        # Normalise numerical features if normalise option is selected
        for i, feature in enumerate(self.numerical_features):
            if feature_options_num == 'normalise':
                scaler = StandardScaler()
                self.X_train[feature] = scaler.fit_transform(self.X_train[feature].values.reshape(-1, 1))
                self.X_test[feature] = scaler.transform(self.X_test[feature].values.reshape(-1, 1))
                self.X_val[feature] = scaler.transform(self.X_val[feature].values.reshape(-1, 1))
            elif feature_options_num[i] == 'bin': # Bin numerical features if bin option is selected
                self.X_train[feature], bin_edges = pd.cut(self.X_train[feature], bins=num_bins, retbins=True)
                self.X_test[feature] = pd.cut(self.X_test[feature], bins=bin_edges)
                self.X_val[feature] = pd.cut(self.X_val[feature], bins=bin_edges)
            else:
                raise ValueError('Invalid value for feature_options. Valid values are \'normalise\', \'bin\', \'woe\', and \'onehot\'')
        for i, feature in enumerate(self.categorical_features):   
            if feature_options_cat == 'woe': # Replace numerical features with weight of evidence values if woe option is selected
                self.X_train[feature] = self.X_train[feature].apply(lambda x: self.woe(x, self.y_train))
                self.X_test[feature] = self.X_test[feature].apply(lambda x: self.woe(x, self.y_train))
                self.X_val[feature] = self.X_val[feature].apply(lambda x: self.woe(x, self.y_train))
            elif feature_options_cat[i] == 'onehot':
                self.X_train = pd.get_dummies(self.X_train, columns=[feature], drop_first=True)
                self.X_test = pd.get_dummies(self.X_test, columns=[feature], drop_first=True)
                self.X_val = pd.get_dummies(self.X_val, columns=[feature], drop_first=True)
            else:
                raise ValueError('Invalid value for feature_options. Valid values are \'normalise\', \'bin\', \'woe\', and \'onehot\'')

    def perform_random_sampling(self):
        # Implement random under and over-sampling to equalize classes for classification
        oversampled = resample(self.X_train[self.y_train == 1], replace=True, n_samples=self.X_train[self.y_train == 0].shape[0], random_state=42)

    def train_classification_models(self):
        # Implement training of different classification models with random sampling
        pass

    def train_regression_models(self):
        # Implement training of different regression models
        pass

    def train_time_series_models(self):
        # Implement training of different time series models
        pass

    def evaluate_models(self):
        # Implement evaluation of all attempted models and store the results in model_history
        pass

    def find_best_model(self):
        # Implement logic to find the best model based on evaluation metrics in model_history
        pass

    def fit(self, dataframe, data_cleaning=False, date_split=False, train_cutoff=None,
            test_cutoff=None, val_cutoff=None, feature_options=None):
        self.load_data(dataframe)

        if data_cleaning:
            self.data_cleaning()

        self.split_data(date_split, train_cutoff, test_cutoff, val_cutoff)

        if feature_options:
            self.feature_engineering(feature_options)

        self.perform_random_sampling()

        self.train_classification_models()
        self.train_regression_models()
        self.train_time_series_models()

        self.evaluate_models()

        self.find_best_model()

    def predict(self, X):
        # Implement prediction using the best model
        pass

    def get_best_model(self):
        # Return the best model object
        pass

    def get_model_history(self):
        # Return the model history with relevant performance metrics
        pass
