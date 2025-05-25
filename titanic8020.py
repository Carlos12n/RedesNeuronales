import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(page_title="Predicción Titanic 80-20", layout="wide")

# Título
st.title("Predicción de Supervivencia en el Titanic (80% Entrenamiento - 20% Validación)")

# Cargar datos
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return None

data = load_data()

if data is not None:
    # Sidebar
    st.sidebar.header("Configuración del Modelo")
    
    # Preprocesamiento
    st.sidebar.subheader("Preprocesamiento")
    balance_method = st.sidebar.selectbox("Balanceo de clases", 
                                        ["Ninguno", "SMOTE", "Submuestreo"])
    feature_engineering = st.sidebar.checkbox("Ingeniería de características", True)

    # Modelo
    st.sidebar.subheader("Arquitectura del Modelo")
    n_layers = st.sidebar.slider("Capas ocultas", 1, 5, 2)
    neurons = st.sidebar.text_input("Neuronas por capa (separadas por coma)", "64,32")
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2)
    activation = st.sidebar.selectbox("Función de activación", ["relu", "tanh", "sigmoid"])

    # Entrenamiento
    st.sidebar.subheader("Entrenamiento")
    epochs = st.sidebar.slider("Épocas", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch size", 16, 128, 32)
    learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.01, 0.001)

    # Preprocesamiento
    def preprocess(df):
        df = df.copy()
        
        if feature_engineering:
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            df['Title'] = df['Title'].replace('Mlle', 'Miss')
            df['Title'] = df['Title'].replace('Ms', 'Miss')
            df['Title'] = df['Title'].replace('Mme', 'Mrs')
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
        if feature_engineering:
            numeric_features.extend(['FamilySize'])
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        categorical_features = ['Sex', 'Embarked']
        if feature_engineering:
            categorical_features.append('Title')
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
        
        X_processed = preprocessor.fit_transform(X)
        
        if balance_method == "SMOTE":
            X_processed, y = SMOTE(random_state=42).fit_resample(X_processed, y)
        elif balance_method == "Submuestreo":
            X_processed, y = RandomUnderSampler(random_state=42).fit_resample(X_processed, y)
            
        return X_processed, y, preprocessor

    # División 80-20
    X, y, preprocessor = preprocess(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo
    def build_model(input_shape):
        model = Sequential()
        model.add(Dense(int(neurons.split(",")[0]), input_dim=input_shape, activation=activation))
        model.add(Dropout(dropout))
        
        for n in neurons.split(",")[1:]:
            model.add(Dense(int(n), activation=activation))
            model.add(Dropout(dropout))
            
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    # Entrenamiento
    model = build_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluación
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    y_pred_prob = model.predict(X_val)
    
    # Métricas
    st.subheader("Resultados en el Conjunto de Validación (20%)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Exactitud", f"{accuracy_score(y_val, y_pred):.4f}")
        st.metric("Precisión", f"{precision_score(y_val, y_pred):.4f}")
        
    with col2:
        st.metric("Sensibilidad", f"{recall_score(y_val, y_pred):.4f}")
        st.metric("F1 Score", f"{f1_score(y_val, y_pred):.4f}")
    
    st.metric("ROC AUC", f"{roc_auc_score(y_val, y_pred_prob):.4f}")

    # Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Validación')
    ax1.set_title('Precisión durante el Entrenamiento')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida durante el Entrenamiento')
    ax2.legend()
    
    st.pyplot(fig)

    # Matriz de Confusión
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    st.pyplot(fig)

    # Curva ROC
    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_val, y_pred_prob):.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Tasa Falsos Positivos')
    ax.set_ylabel('Tasa Verdaderos Positivos')
    ax.legend()
    st.pyplot(fig)

    # Guardar modelo
    if st.button("Guardar Modelo"):
        model.save("modelo_titanic_80_20.h5")
        st.success("Modelo guardado como 'modelo_titanic_80_20.h5'")
else:
    st.error("No se pudieron cargar los datos")