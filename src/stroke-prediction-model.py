import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from skl2onnx import to_onnx 
from mlprodict.onnxrt import OnnxInference
from onnxruntime import InferenceSession

def getPipeline():
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())])

    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type',
            'smoking_status']
    categorical_transformer = Pipeline([
        ('ohe', OneHotEncoder(drop='first', handle_unknown = 'ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categories', categorical_transformer, categorical_features)],
        remainder='passthrough'
    )
    
    pipe = Pipeline(
    steps = [
        ('preprocess', preprocessor),
        ('clf', LogisticRegression(C=1, class_weight="balanced", penalty="l2"))
    ])
    return pipe


df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")

# Target column
y = df['stroke']

# Matrix
X = df
X.drop('stroke', axis=1, inplace=True) 
X.drop('id', axis=1, inplace=True) 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Init pipeline
pipeline = getPipeline()

# Fit the pipeline to the train data
pipeline.fit(X_train, y_train)

def predict(X):
    print(pipeline.predict(X))
    
X = pd.DataFrame({
    'gender': ['Male'],
    'age': [12.0],
    'hypertension': [0],
    'heart_disease': [1],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [228.69],
    'bmi': [36.6],
    'smoking_status': ['formerly smoked']
})

predict(X)