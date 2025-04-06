import pandas as pd
import numpy as np
import joblib  # For saving the model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics

# Load datasets
train = pd.read_csv("datasets/Train_data_1.csv")
test = pd.read_csv("datasets/Test_data_1.csv")

# Remove redundant column
train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Standardize numerical features
scaler = StandardScaler()

# Save the scaler after fitting
joblib.dump(scaler, "scaler.pkl")
num_cols = train.select_dtypes(include=['float64', 'int64']).columns

train_scaled = scaler.fit_transform(train[num_cols])
test_scaled = scaler.transform(test[num_cols])

train_scaled_df = pd.DataFrame(train_scaled, columns=num_cols)
test_scaled_df = pd.DataFrame(test_scaled, columns=num_cols)

# Encode categorical features
encoder = LabelEncoder()
cat_train = train.select_dtypes(include=['object']).copy()
cat_test = test.select_dtypes(include=['object']).copy()

# Ensure 'class' exists in training data before processing
if 'class' in cat_train.columns:
    y_train = encoder.fit_transform(cat_train['class'])  # Encode class labels
    cat_train.drop(['class'], axis=1, inplace=True)  # Remove class from training features
else:
    raise ValueError("Error: 'class' column is missing from training data")

# Apply LabelEncoder to categorical columns
for col in cat_train.columns:
    encoder.fit(cat_train[col])  # Fit on train data
    cat_train[col] = encoder.transform(cat_train[col])  # Encode train data
    if col in cat_test.columns:
        cat_test[col] = cat_test[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# Prepare final datasets
train_x = pd.concat([train_scaled_df, cat_train], axis=1)
train_y = y_train  # Encoded class labels

# Ensure categorical columns are handled correctly in test data
test_x = pd.concat([test_scaled_df, cat_test], axis=1)

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, train_size=0.70, random_state=2)

# Save feature names before training
joblib.dump(list(X_train.columns), "feature_names.pkl")

# Train Decision Tree Classifier
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)

# Save the trained model
joblib.dump(DTC_Classifier, 'decision_tree_model.pkl')

# During training, save the encoder
joblib.dump(encoder, 'encoder.pkl')

# Load the saved model
loaded_model = joblib.load('decision_tree_model.pkl')

# Model Evaluation on Validation Data
accuracy_val = metrics.accuracy_score(Y_val, loaded_model.predict(X_val))
conf_matrix_val = metrics.confusion_matrix(Y_val, loaded_model.predict(X_val))
classification_val = metrics.classification_report(Y_val, loaded_model.predict(X_val))

# Print Validation Results
print("\n====== Decision Tree Classifier Model Evaluation (Validation Data) ======")
print(f"Accuracy: {accuracy_val}\n")
print("Confusion Matrix:\n", conf_matrix_val)
print("\nClassification Report:\n", classification_val)

