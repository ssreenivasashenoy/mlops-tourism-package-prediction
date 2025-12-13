import os
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

HF_USERNAME = "ShenoySreenivas"
DATASET_NAME = "tourism-data-processed"
MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Tourism-Package-Prediction-Experiment")

# Download processed train and test datasets from Hugging Face
train_path = hf_hub_download(
    repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
    filename="train.csv",
    repo_type="dataset"
)

test_path = hf_hub_download(
    repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
    filename="test.csv",
    repo_type="dataset"
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

# Define models and hyperparameters
models = {
    "RandomForest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    ),
    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        ),
        {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    )
}

best_model = None
best_model_name = None
best_f1 = -1

with mlflow.start_run():

    for model_name, (model, params) in models.items():
        grid = GridSearchCV(
            model,
            params,
            cv=3,
            scoring="f1",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        predictions = grid.best_estimator_.predict(X_test)
        f1 = f1_score(y_test, predictions)

        mlflow.log_metric(f"{model_name}_f1", f1)

        for param_name, param_value in grid.best_params_.items():
            mlflow.log_param(f"{model_name}_{param_name}", param_value)

        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_
            best_model_name = model_name

    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_f1_score", best_f1)

    model_file = "best_tourism_model.pkl"
    joblib.dump(best_model, model_file)
    mlflow.log_artifact(model_file)

    # Register model in Hugging Face Model Hub
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

print(f"Best Model: {best_model_name}")
print(f"Best F1 Score: {best_f1}")
