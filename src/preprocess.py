from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# load data: return a pandas DataFrame
def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    # the marketing_campaign dataset uses ';' as the delimiter
    return pd.read_csv(path, sep=";") 


def make_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Return a preprocessing pipeline plus feature matrix X and target vector y."""

    # split X and y, and exclude any customer identifier column from X
    y = df[target_col]  # target vector

    # Determine columns to exclude
    exclude_cols = [target_col]
    for col in ("ID",):
        if col in df.columns:
            exclude_cols.append(col)

    X = df.drop(columns=exclude_cols)  # feature matrix without ID columns

    # Detect column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Define pipelines: strategies for handling numeric values and categorical values

    # 1. numeric_pipe: impute missing values with median and scale the numeric features
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()), # scale the numeric features
        ]
    )

    # 2. categorical_pipe: impute missing values with most frequent and one-hot encode the categorical features
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
             # convert categorical values to vectors
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # preprocessor: combine the numeric_pipe and categorical_pipe
    preprocessor = ColumnTransformer(
        transformers=[
            ("number", numeric_pipe, num_cols),
            ("category", categorical_pipe, cat_cols),
        ],
        verbose_feature_names_out=False  
    )

    return preprocessor, X, y


# split data into train and test sets: 20% for test set
def split_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split X and y into train/test subsets with stratification when possible."""
    # avoid stratify for regression/unique labels
    stratify = y if y.nunique() < len(y) else None 
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
