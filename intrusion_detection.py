import streamlit as st
import joblib
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Dummy credentials (Replace with a proper authentication system)
USER_CREDENTIALS = {"admin": "admin", "user": "securepass"}

# SMTP Email Configuration (Update with your details)
SMTP_SERVER = "smtp.gmail.com"  # Example: Gmail SMTP
SMTP_PORT = 587  # Standard port for TLS
SENDER_EMAIL = "gopikrishna2934@gmail.com"
SENDER_PASSWORD = "ycio uljj wtnz qdkx"
RECIPIENT_EMAIL = "gopikrishna2001215@gmail.com"  # Where alerts will be sent

# Function to send email alert
def send_email_alert(attack_df):
    subject = "🚨 Intrusion Detection Alert: Attacks Detected"
    
    # Prepare attack details
    attack_info = attack_df.to_string(index=False)
    body = f"""
    ALERT! 🚨

    Intrusion detection system has detected {len(attack_df)} attack(s).

    Details:
    {attack_info}

    Please take necessary action immediately.
    """

    # Set up email message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to SMTP server and send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        st.success("📧 Alert email sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

# Function to check login credentials
def login():
    st.title("🔐 User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password", key="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password!")

# Function to load and process data
def load_and_predict():
    st.title("Intrusion Detection System - Batch Prediction")
    st.write(f"Welcome, **{st.session_state['username']}** 👋")

    # File Upload Section
    uploaded_file = st.file_uploader("📂 Upload a CSV file for prediction", type=["csv"])
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        
        # Load the trained model
        model = joblib.load("decision_tree_model.pkl")

        # Load the saved encoder
        encoder = joblib.load("encoder.pkl")  

        # Load feature names used during training
        expected_features = joblib.load("feature_names.pkl")  

        # Read the uploaded CSV file
        test_data = pd.read_csv(uploaded_file)

        # Drop extra columns not used during training
        test_data = test_data[[col for col in expected_features if col in test_data.columns]]

        # Encode categorical columns safely
        categorical_columns = ["protocol_type", "service", "flag"]
        for col in categorical_columns:
            if col in test_data.columns:
                test_data[col] = test_data[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

        # Make predictions
        predictions = model.predict(test_data)

        # Map numerical predictions to labels
        label_mapping = {0: "Normal", 1: "Attack"}
        test_data["Prediction"] = [label_mapping[pred] for pred in predictions]

        # Filter only "Attack" rows
        attack_df = test_data[test_data["Prediction"] == "Attack"]

        # Display only attack records
        if attack_df.empty:
            st.success("✅ No attacks detected in the dataset!")
        else:
            attack_df = attack_df.head(100)
            st.subheader("🚨 Detected Attacks")
            # st.dataframe(attack_df)

            # Send an email alert
            send_email_alert(attack_df)

        # Convert attack records to CSV for download
        csv = attack_df.to_csv(index=False)
        st.download_button("📥 Download Attack Data", csv, "attacks.csv", "text/csv")

# Main logic: Check login state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    load_and_predict()
