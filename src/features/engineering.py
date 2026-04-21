import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SERVICE_COLS_RAW = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]


def preprocess(df):
    df = df.copy()

    df = df.drop(columns=["customerID"], errors="ignore")

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    for col in YES_NO_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    df["gender"] = (df["gender"] == "Male").astype(int)

    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)

    return df


def build_features(df):
    df = df.copy()

    # Ratio entre lo que ha pagado y lo que deberia haber pagado segun tenure
    expected = df["tenure"] * df["MonthlyCharges"]
    df["charge_ratio"] = df["TotalCharges"] / (expected + 1)

    # Numero de servicios adicionales contratados
    service_cols = [c for c in df.columns if any(s in c for s in SERVICE_COLS_RAW)]
    df["num_services"] = df[service_cols].sum(axis=1)

    return df


def split_data(df, target="Churn", test_size=0.2, random_state=261):
    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"Train: {X_train_sc.shape[0]} muestras | Churn: {y_train.mean()*100:.1f}%")
    print(f"Test:  {X_test_sc.shape[0]} muestras  | Churn: {y_test.mean()*100:.1f}%")

    return X_train_sc, X_test_sc, y_train, y_test, scaler
