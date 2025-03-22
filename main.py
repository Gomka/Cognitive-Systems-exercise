import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Encode categorical variables
label_encoders = {}
for col in ['gender', 'smoking_history']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for later decoding if needed

# Define features and target
X = df.drop(columns=['diabetes'])  # Features
y = df['diabetes']  # Target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Next we use the model to compute the probability of diabetes of a new user

# Define new user input (example data)
new_user = {
    "gender": "Female",
    "age": 80.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 48.5,
    "HbA1c_level": 5.9,
    "blood_glucose_level": 220
}

# Convert categorical variables using the same encoding as before
new_user["gender"] = label_encoders["gender"].transform([new_user["gender"]])[0]
new_user["smoking_history"] = label_encoders["smoking_history"].transform([new_user["smoking_history"]])[0]

# Convert dictionary to DataFrame for consistency
new_user_df = pd.DataFrame([new_user])

# Scale numerical features using the same scaler as before
new_user_scaled = scaler.transform(new_user_df)

# Make a prediction
new_prediction = model.predict(new_user_scaled)[0]
new_probability = model.predict_proba(new_user_scaled)[0][1]  # Probability of diabetes

# Print the result
print(f"Predicted Diabetes Status: {'Diabetic' if new_prediction == 1 else 'Non-Diabetic'}")
print(f"Probability of Having Diabetes: {new_probability:.2f}")
