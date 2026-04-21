import pandas as pd
from pathlib import Path


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    df = pd.read_csv(path)

    # TotalCharges viene como string cuando tenure == 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    return df


def validate_data(df):
    report = {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "churn_distribution": df["Churn"].value_counts().to_dict(),
        "churn_rate_pct": round((df["Churn"] == "Yes").mean() * 100, 2),
    }

    print(f"Dataset: {report['shape'][0]} filas x {report['shape'][1]} columnas")
    print(f"Churn rate: {report['churn_rate_pct']}%")
    print(f"Duplicados: {report['duplicated_rows']}")
    print(f"Nulos totales: {sum(report['null_counts'].values())}")

    return report
