# MLToolkit

The AutomatedMachineLearning class is designed to provide an automated pipeline for performing machine learning tasks on a given dataset. It aims to handle various data preprocessing tasks, model training, evaluation, and selection of the best model. The class provides methods to load data, perform data cleaning, split data, feature engineering, train different models, and evaluate their performance.

## Usage

To use the AutomatedMachineLearning class, follow these steps:

- Import the necessary libraries and the AutomatedMachineLearning class:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
# Add other models as needed
```
- Instantiate the AutomatedMachineLearning class with the target variable name:

```python
target_variable = 'target'  # Replace 'target' with the name of your target variable
automl = AutomatedMachineLearning(target_variable)
```
- Load your dataset into the class:

```python
data = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with the path to your dataset
automl.load_data(data)
```
- Configure the options for data cleaning, data splitting, and feature engineering (optional):

``` python
# Example of data cleaning and splitting based on dates:
automl.data_cleaning()
automl.split_data(date_col='date_column', date_split=True, train_cutoff_date='2023-01-01', test_cutoff_date='2023-06-30')

# Example of feature engineering options:
feature_options_num = ['norm', 'bin', 'norm']  # Replace with options for each numerical feature
feature_options_cat = 'onehot'  # Replace with options for each categorical feature
automl.feature_engineering(feature_options_num, feature_options_cat, num_bins=10)
```
- Train the models and find the best model:

```python
automl.fit(data_cleaning=True, date_split=True, train_cutoff='2023-01-01', test_cutoff='2023-06-30', feature_options_num=feature_options_num, feature_options_cat=feature_options_cat)
best_model = automl.get_best_model()
```
- Make predictions using the best model:

``` python
X_new_data = pd.DataFrame(...)  # Replace with your new data to predict
predictions = automl.predict(X_new_data)
```

- Access the model history and performance metrics:

``` python
model_history = automl.get_model_history()
print(model_history)
```

## Class Methods
```load_data(self, dataframe)```

Load data from a pandas DataFrame into the class.
```data_cleaning(self, ...) (Optional)```

Perform data cleaning operations on the loaded DataFrame, such as handling missing values, outliers, duplicates, etc.
```split_data(self, ...) (Optional)```

Split the dataset into training, test, and validation sets based on specified options.
```feature_engineering(self, ...) (Optional)```

Preprocess features based on the specified options, including normalizing, one-hot encoding, weight of evidence transformation, or binning.
```perform_random_sampling(self)```

Implement random under and over-sampling to equalize classes for classification tasks.
```train_classification_models(self), train_regression_models(self), train_time_series_models(self)```

Implement training of different classification, regression, and time series models, respectively.
```evaluate_models(self)```

Implement evaluation of all attempted models and store the results in model_history.
```find_best_model(self)```

Implement logic to find the best model based on evaluation metrics in model_history.
```fit(self, ...)```

The main method that orchestrates the entire pipeline. Loads data, performs data cleaning, feature engineering, and model training, evaluates and selects the best model.
```predict(self, X)```

Implement prediction using the best model.
```get_best_model(self)```

Return the best model object.
```get_model_history(self)```

Return the model history with relevant performance metrics.
Note

Please note that this implementation serves as a basic template for an automated machine learning pipeline. Depending on the specific requirements of your dataset and tasks, you may need to modify and extend the class methods accordingly. Additionally, more models and customizations can be added to the class as needed.

Ensure that you have the necessary libraries and dependencies installed before using the AutomatedMachineLearning class.