import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn

# Setup MLflow
mlflow.set_experiment("attrition_prediction")
mlflow.set_tracking_uri("file:./mlruns")  # Explicit tracking URI

print("Loading data...")
df = pd.read_csv('data/raw/attrition_data.csv')
df.attrition = df.attrition.map(dict(Yes = 1, No = 0))

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

# Prepare features
X = df.drop('attrition', axis=1)
y = df['attrition']

# Handle categorical columns
print("\nProcessing features...")
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {len(categorical_cols)}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Remove constant columns
X = X.loc[:, X.nunique() > 1]
print(f"Final features: {X.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Attrition rate - Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

# Start MLflow run
with mlflow.start_run(run_name="random_forest_v1") as run:
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    # Log all parameters
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("test_size", 0.2)
    
    # Train model
    model = RandomForestClassifier(**params, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✅ Training complete")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for name, value in metrics.items():
        print(f"{name:15s}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save locally first
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print("\n" + "="*60)
    print("LOGGING TO MLFLOW")
    print("="*60)
    
    # Log the model to MLflow (THIS IS THE KEY PART!)
    print("Logging model...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # This creates a "model" folder in artifacts
        registered_model_name="attrition_model",  # Register in Model Registry
        input_example=X_train.iloc[:5],  # Add example input
    )
    print("✅ Model logged to MLflow")
    
    # Log additional artifacts
    print("Logging artifacts...")
    mlflow.log_artifact('models/feature_importance.csv')
    mlflow.log_artifact('models/label_encoders.pkl')
    mlflow.log_artifact('models/feature_names.pkl')
    print("✅ Artifacts logged")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered as: attrition_model")
    print(f"Local model saved: models/model.pkl")
    print(f"\n🔗 View in MLflow UI: mlflow ui")
    print("="*60)

print("\n✅ ALL DONE!")
print("\nNext steps:")
print("1. Run: mlflow ui")
print("2. Open: http://localhost:5000")
print("3. Check 'Models' tab for 'attrition_model'")
print("4. Click on your run to see artifacts (you should see 'model' folder)")