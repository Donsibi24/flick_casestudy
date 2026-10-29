import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

student_data = pd.read_csv('student-data (3).csv')


# seperate x and y
x = student_data.drop(['passed'], axis=1)
y = student_data['passed']

#Train and split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state=42)

cate_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'romantic',]
num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

# Create the preprocessing pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cate_cols),
        ('num', StandardScaler(), num_cols)
    ]
)



# preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

#Fit and trnsform training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_processed = preprocessing_pipeline.fit_transform(x_train)
x_test_processed  = preprocessing_pipeline.transform(x_test)

#Logistic Regression model
model =LogisticRegression(max_iter=1000)
model.fit(x_train_processed,y_train)
y_pred_model=model.predict(x_test_processed)
print("Model Accuracy:", model.score(x_test_processed, y_test))

#save the model

pickle.dump(model, open('model.pkl', 'wb'))

print('Model training completed and saved as model.pkl')
