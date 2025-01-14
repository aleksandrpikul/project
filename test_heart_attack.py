# -*- coding: utf-8 -*-
"""Heart Attack Prediction: Testing and Model Implementation"""

# Импорт библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pytest

# Глобальный путь к данным
DATA_PATH = 'heart_attack_youth_adult_france.csv'

# Функция для загрузки данных
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Функция для предобработки данных
def preprocess_data(df):
    df.drop('Patient_ID', axis=1, inplace=True)
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

# Функция для разделения данных
def split_data(df):
    X = df.drop('Heart_Attack', axis=1)
    y = df['Heart_Attack']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Функция для обучения модели
def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Тест загрузки данных
def test_data_loading():
    df = load_data()
    assert not df.empty, "Данные не загружены или пустые."
    required_columns = ['Patient_ID', 'Heart_Attack']
    for col in required_columns:
        assert col in df.columns, f"Отсутствует столбец: {col}"

# Тест предобработки данных
def test_data_preprocessing():
    df = load_data()
    df = preprocess_data(df)
    assert 'Patient_ID' not in df.columns, "Столбец Patient_ID не удален."
    assert df.isnull().sum().sum() == 0, "В данных есть пропущенные значения."

# Тест разделения данных
def test_data_splitting():
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0 and len(X_test) > 0, "Данные не разделены."
    assert len(X_train) == len(y_train), "Несоответствие между X_train и y_train."

# Тест обучения модели
def test_model_training():
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    accuracy = train_model(X_train, X_test, y_train, y_test)
    assert accuracy > 0.5, f"Точность модели слишком низкая: {accuracy}"
