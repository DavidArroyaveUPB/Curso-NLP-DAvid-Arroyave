import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from sklearn.model_selection import train_test_split




# Usar su ID UPB Ejemplo: "0028984798"
id_upb = "000573992"

data_reviews = pd.read_parquet(
    "https://www.dropbox.com/scl/fi/gvk9yj8cn96oocr9z058x/filmaffinity_reviews_dataset.parquet?rlkey=xgvr00zvkxbkwqqavqutpsshg&st=xjb7xze9&dl=1"
)
data_reviews = data_reviews.sample(n=50_000, random_state=int(id_upb))
data_reviews.info()

data_reviews.sample(5)

data_reviews.isnull().sum()


data_reviews.duplicated().sum()


UMBRAL = 6
data_reviews["sentiment_bin"] = (data_reviews["author_rating"] > UMBRAL).astype(int) # Dividir los datos en conjuntos de entrenamiento y prueba

X_data = data_reviews["author_review_desc"]
y_data = data_reviews["sentiment_bin"]
X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.2,
    stratify=y_data,  # Mantener la proporción de clases en ambos conjuntos
    random_state=42,
)
## 1) Limpieza de los datos de texto
X_train = X_train.applymap(lambda x: x.lower() if isinstance(x, str) else x)
X_train = X_train.applymap(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x) if isinstance(x, str) else x)
X_Train = X_Train.applymap(
    lambda x: str(TextBlob(x).correct()) if isinstance(x, str) else x
)
from sklearn.feature_extraction.text import TfidfVectorizer
## 2) Representacion del texto
Luego de tener los datos limpios, realizar la representación de los datos de texto para poder usarse en modelos de machine learning.

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_Train['columna_texto'])

print("Shape:", X_train_tfidf.shape)
## 3) Entrenar un modelo de machine learning de clasificación, Utilizar un modelo de clasificación para entrenar y evaluar el modelo con los datos preparados.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_Train, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4) Evaluar el modelo

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
