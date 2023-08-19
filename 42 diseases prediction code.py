import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
train_data = pd.read_csv("Training.csv").dropna(axis=1)

# Encoding the target value into numerical value
encoder = LabelEncoder()
train_data["prognosis"] = encoder.fit_transform(train_data["prognosis"])

# Prepare the training data
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Training the SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Training the Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Training the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)

# Save the trained models and encoder using pickle
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

import pandas as pd
import pickle
from scipy.stats import mode

# Load the saved models and encoder using pickle
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

def preprocess_input(input_data):
    # Preprocessing steps
    return input_data

# Function to make disease predictions
# Function to make disease predictions
def predict_disease(input_data, encoder):
    # Preprocess the input data
    input_data = preprocess_input(input_data)

    # Make predictions using the loaded models
    svm_prediction = svm_model.predict(input_data)
    nb_prediction = nb_model.predict(input_data)
    rf_prediction = rf_model.predict(input_data)

    # Combine the predictions (use mode, voting, or any other desired method)
    final_prediction = mode([svm_prediction, nb_prediction, rf_prediction]).mode[0][0]

    # Convert the numerical prediction to disease name using the label encoder
    disease_name = encoder.inverse_transform([final_prediction])[0]

    return disease_name

# Example usage
input_data = pd.DataFrame([[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])  # Replace with your input data
prediction = predict_disease(input_data, encoder)
print("Predicted Disease:", prediction)

