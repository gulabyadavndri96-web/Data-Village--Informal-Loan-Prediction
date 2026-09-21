
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_regression_model.sav')

# Define the expected feature columns (in the correct order as used during training)
# This is crucial for consistent input to the model
expected_columns = [
    'Age', 'Farming_experience_years', 'Smartphone_access',
    'Financial_literacy_score', 'Bank_accessibility',
    'Education_Primary', 'Education_SSC',
    'Land_ownership_Medium farmer', 'Land_ownership_Small farmer',
    'Main_source_of_income_Dairy', 'Main_source_of_income_Farming',
    'Main_source_of_income_Service', 'Main_source_of_income_Wage labour'
]

st.title('Informal Loan Prediction App')
st.write('Enter the individual\'s details to predict if they are likely to have taken an informal loan.')

# Input fields for numerical features
age = st.number_input('Age', min_value=18, max_value=100, value=30)
farming_experience = st.number_input('Farming Experience (years)', min_value=0, max_value=70, value=10)
smartphone_access = st.selectbox('Smartphone Access', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
financial_literacy = st.number_input('Financial Literacy Score', min_value=0, max_value=10, value=5)
bank_accessibility = st.selectbox('Bank Accessibility', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Input fields for categorical features (matching the original categories and encoding logic)
education_options = ['High School', 'Primary', 'SSC']
education_selected = st.selectbox('Education', education_options)

land_ownership_options = ['Large farmer', 'Medium farmer', 'Small farmer']
land_ownership_selected = st.selectbox('Land Ownership', land_ownership_options)

main_source_of_income_options = ['Business', 'Dairy', 'Farming', 'Service', 'Wage labour']
main_source_of_income_selected = st.selectbox('Main Source of Income', main_source_of_income_options)


# Process inputs for prediction
if st.button('Predict'):
    # Initialize all one-hot encoded columns to False
    input_data = {
        'Age': age,
        'Farming_experience_years': farming_experience,
        'Smartphone_access': smartphone_access,
        'Financial_literacy_score': financial_literacy,
        'Bank_accessibility': bank_accessibility,
        'Education_Primary': False,
        'Education_SSC': False,
        'Land_ownership_Medium farmer': False,
        'Land_ownership_Small farmer': False,
        'Main_source_of_income_Dairy': False,
        'Main_source_of_income_Farming': False,
        'Main_source_of_income_Service': False,
        'Main_source_of_income_Wage labour': False
    }

    # Set relevant one-hot encoded columns based on user selection (matching drop_first=True logic)
    if education_selected == 'Primary':
        input_data['Education_Primary'] = True
    elif education_selected == 'SSC':
        input_data['Education_SSC'] = True

    if land_ownership_selected == 'Medium farmer':
        input_data['Land_ownership_Medium farmer'] = True
    elif land_ownership_selected == 'Small farmer':
        input_data['Land_ownership_Small farmer'] = True

    if main_source_of_income_selected == 'Dairy':
        input_data['Main_source_of_income_Dairy'] = True
    elif main_source_of_income_selected == 'Farming':
        input_data['Main_source_of_income_Farming'] = True
    elif main_source_of_income_selected == 'Service':
        input_data['Main_source_of_income_Service'] = True
    elif main_source_of_income_selected == 'Wage labour':
        input_data['Main_source_of_income_Wage labour'] = True

    # Create DataFrame from input_data, ensuring column order matches training data
    input_df = pd.DataFrame([input_data], columns=expected_columns)

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[:, 1][0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.success(f"The individual is likely to have taken an informal loan. (Probability: {prediction_proba:.2f})")
    else:
        st.info(f"The individual is unlikely to have taken an informal loan. (Probability: {prediction_proba:.2f})")

st.write("\nTo run this app:\n1. Save this code as `streamlit_app.py` and the model as `logistic_regression_model.sav` in the same directory.\n2. Open your terminal, navigate to that directory, and run `streamlit run streamlit_app.py`.")
