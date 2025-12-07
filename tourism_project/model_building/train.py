<<<<<<< HEAD
import os
import pandas as pd
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow Setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tourism-Package-Prediction-Experiment")

# HF API Setup
HF_USERNAME = "ShenoySreenivas"
DATASET_NAME = "tourism-data-processed"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Download Processed Data
X_train_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="X_train.csv", repo_type="dataset")
X_test_path  = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="X_test.csv", repo_type="dataset")
y_train_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="y_train.csv", repo_type="dataset")
y_test_path  = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="y_test.csv", repo_type="dataset")

# Load into DataFrames
X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).values.ravel()
y_test  = pd.read_csv(y_test_path).values.ravel()

# Preprocessing
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Model Dictionary
models = {
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "randomforestclassifier__n_estimators": [100, 200],
            "randomforestclassifier__max_depth": [5, 10, None]
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "xgbclassifier__n_estimators": [50, 100, 150],
            "xgbclassifier__max_depth": [2, 4, 6],
            "xgbclassifier__learning_rate": [0.01, 0.05, 0.1]
        }
    )
}

best_model, best_name, best_f1 = None, None, -1

# Training & MLflow Logging
with mlflow.start_run():

    for name, (algo, params) in models.items():
        pipeline = make_pipeline(preprocessor, algo)
        grid = GridSearchCV(pipeline, params, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        score = f1_score(y_test, grid.best_estimator_.predict(X_test))

        mlflow.log_metric(f"{name}_f1", score)
        mlflow.log_params(grid.best_params_)

        if score > best_f1:
            best_f1, best_model, best_name = score, grid.best_estimator_, name

    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_f1_score", best_f1)

    # Save Model
    model_file = "best_tourism_model.pkl"
    joblib.dump(best_model, model_file)
    mlflow.log_artifact(model_file)

    # Upload to HF Model Hub
    MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

    try:
        api.repo_info(MODEL_REPO, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_file,
        path_in_repo=model_file,
        repo_id=MODEL_REPO,
        repo_type="model",
        create_pr=False
    )

print(f"Best Model: {best_name} | F1 Score: {best_f1:.4f}")
=======
import os
import pandas as pd
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow Setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tourism-Package-Prediction-Experiment")

# HF API Setup
HF_USERNAME = "ShenoySreenivas"
DATASET_NAME = "tourism-data-processed"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Download Processed Data
X_train_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="X_train.csv", repo_type="dataset")
X_test_path  = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="X_test.csv", repo_type="dataset")
y_train_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="y_train.csv", repo_type="dataset")
y_test_path  = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="y_test.csv", repo_type="dataset")

# Load into DataFrames
X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).values.ravel()
y_test  = pd.read_csv(y_test_path).values.ravel()

# Preprocessing
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Model Dictionary
models = {
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "randomforestclassifier__n_estimators": [100, 200],
            "randomforestclassifier__max_depth": [5, 10, None]
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "xgbclassifier__n_estimators": [50, 100, 150],
            "xgbclassifier__max_depth": [2, 4, 6],
            "xgbclassifier__learning_rate": [0.01, 0.05, 0.1]
        }
    )
}

best_model, best_name, best_f1 = None, None, -1

# Training & MLflow Logging
with mlflow.start_run():

    for name, (algo, params) in models.items():
        pipeline = make_pipeline(preprocessor, algo)
        grid = GridSearchCV(pipeline, params, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        score = f1_score(y_test, grid.best_estimator_.predict(X_test))

        mlflow.log_metric(f"{name}_f1", score)
        mlflow.log_params(grid.best_params_)

        if score > best_f1:
            best_f1, best_model, best_name = score, grid.best_estimator_, name

    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_f1_score", best_f1)

    # Save Model
    model_file = "best_tourism_model.pkl"
    joblib.dump(best_model, model_file)
    mlflow.log_artifact(model_file)

    # Upload to HF Model Hub
    MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

    try:
        api.repo_info(MODEL_REPO, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_file,
        path_in_repo=model_file,
        repo_id=MODEL_REPO,
        repo_type="model",
        create_pr=False
    )

print(f"Best Model: {best_name} | F1 Score: {best_f1:.4f}")
>>>>>>> ab18067c2ed4da85e56bb22f8aee52f6970252a7
