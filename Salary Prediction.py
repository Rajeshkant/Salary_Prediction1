import streamlit as st
import joblib

# Load the model, LabelEncoder, and fitted ColumnTransformer from the saved files
loaded_model = joblib.load(r'Salary_Prediction.sav')
loaded_label_encoder = joblib.load(r'LabelEncoder.pkl')
loaded_column_transformer = joblib.load(r'ColumnTransformer.pkl')

# Creating a function for prediction
def Salary_Prediction(input_experience, input_job_role):
    # Convert user input job role to numerical using the loaded LabelEncoder
    input_job_role_encoded = loaded_label_encoder.transform([input_job_role])

    # Apply one-hot encoding to user input job role using the fitted ColumnTransformer
    input_features = loaded_column_transformer.transform([[input_experience, input_job_role_encoded[0]]])

    # Making prediction for the user input using the loaded model
    predicted_salary = loaded_model.predict(input_features)

    return predicted_salary[0]

def main():
    # Giving a Title
    st.title('Salary Prediction')

    # Getting the Input data From the user
    years_of_experience = st.text_input('Years of Experience')
    job_role = st.text_input('Job Role')

    # Code for Prediction
    predicted_salary = ""
    
    # Creating Examine for prediction
    if st.button('Predict Salary'):
        predicted_salary = Salary_Prediction(float(years_of_experience), job_role)
        
    st.success(f"Predicted Salary for {years_of_experience} years of experience and job role {job_role}: {predicted_salary}")
    
if __name__=='__main__':
    main()


