import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load the dataset
data = pd.read_csv('/content/training.csv')

# Check for missing values and handle them if necessary
data.fillna('None', inplace=True)  # Example: replace NaN with 'None' for categorical features

# Preprocessing
label_encoders = {}
categorical_cols = ['Gender', 'Lifestyle Type', 'Allergies', 
                    'Medical Conditions', 'Stress Levels', 'Diet Type', 
                    'Physical Limitations']

# Encode categorical columns for features
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Encode the 'Goal' column separately
goal_encoder = LabelEncoder()
data['Goal'] = goal_encoder.fit_transform(data['Goal'])

# Prepare features and labels
X = data[['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Goal', 
           'Lifestyle Type', 'Allergies', 'Medical Conditions', 
           'Stress Levels', 'Diet Type', 'Physical Limitations']]

# Create label encoders for meal choices
meal_encoders = {
    'Breakfast Choices': LabelEncoder(),
    'Lunch Choices': LabelEncoder(),
    'Dinner Choices': LabelEncoder()
}

# Fit the meal encoders
for meal in meal_encoders.keys():
    data[meal] = meal_encoders[meal].fit_transform(data[meal])

# Convert the labels into a DataFrame
y = pd.DataFrame()
y['Breakfast Choices'] = data['Breakfast Choices'] 
y['Lunch Choices'] = data['Lunch Choices'] 
y['Dinner Choices'] = data['Dinner Choices'] 
y['Exercise Frequency (days/week)'] = data['Exercise Frequency (days/week)'].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='linear')  # Output layer 
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Use 'mse' for regression

# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Function to recommend meals and exercise based on dynamic user inputs
def recommend_meals_and_exercise(user_input):
    # Convert user_input to DataFrame
    input_df = pd.DataFrame(user_input, index=[0])

    # Preprocess the input using the stored label encoders
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col]) 

    # Encode the 'Goal'
    input_df['Goal'] = goal_encoder.transform(input_df['Goal'])

    # Make predictions
    predictions = model.predict(input_df)

    # Process predictions
    meal_choices = {
        "Recommended Breakfast": meal_encoders['Breakfast Choices'].inverse_transform([int(predictions[0, 0])])[0],
        "Recommended Lunch": meal_encoders['Lunch Choices'].inverse_transform([int(predictions[0, 1])])[0],
        "Recommended Dinner": meal_encoders['Dinner Choices'].inverse_transform([int(predictions[0, 2])])[0]
    }
    exercise_frequency = int(predictions[0, 3])

    return {
        **meal_choices,
        "Recommended Exercise Frequency (days/week)": exercise_frequency
    }

# Function to get user input
def get_user_input():
    user_data = {
        "Age": int(input("Enter your age: ")),
        "Gender": input("Enter your gender (Male/Female/Other): "),
        "Height (cm)": int(input("Enter your height in cm: ")),
        "Weight (kg)": int(input("Enter your weight in kg: ")),
        "Goal": input("Enter your goal (Weight Loss/Muscle Gain/Maintenance): "),
        "Lifestyle Type": input("Enter your lifestyle type (Sedentary/Very Active/Active): "),
        "Allergies": input("Enter any allergies (None/Lactose/Peanuts/Gluten): "),
        "Medical Conditions": input("Enter any medical conditions (Hypertension/Diabetes/Asthma/None): "),
        "Stress Levels": input("Enter your stress levels (Low/Moderate/High): "),
        "Diet Type": input("Enter your diet type (Vegetarian/Balanced/Keto/Vegan): "),
        "Physical Limitations": input("Enter any physical limitations (None/Back Pain/Knee Pain): ")
    }
    return user_data

# Main execution
if __name__ == "__main__":
    user_input = get_user_input()
    recommendations = recommend_meals_and_exercise(user_input)
    print("\nRecommendations:")
    print(recommendations)
