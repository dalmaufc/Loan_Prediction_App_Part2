import streamlit as st
import joblib

# Load the trained models
rf_model_path = "random_forest.pkl"
xgb_model_path = "xgboost.pkl"

rf_classifier = joblib.load(rf_model_path)
xgb_classifier = joblib.load(xgb_model_path)

@st.cache_data()
def prediction(model_choice, Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
    # Pre-processing user input    
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1  
    
    # Ensure consistency with model training (No division by 1000)
    LoanAmount = LoanAmount  
    
    # Select the model
    classifier = rf_classifier if model_choice == "Random Forest" else xgb_classifier
    
    # Making predictions 
    prediction = classifier.predict([[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
    return 'Approved' if prediction == 1 else 'Rejected'

# Main Streamlit app
def main():       
    # Frontend design
    html_temp = """ 
    <div style="background-color:yellow;padding:13px"> 
    <h1 style="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True) 
    
    # Model selection
    model_choice = st.selectbox("Choose a Model", ("Random Forest", "XGBoost"))
    
    # User input fields
    Gender = st.selectbox('Gender',("Male", "Female"))
    Married = st.selectbox('Marital Status',("Unmarried", "Married")) 
    ApplicantIncome = st.number_input("Applicant's Monthly Income", min_value=0)
    LoanAmount = st.number_input("Total Loan Amount", min_value=0)
    Credit_History = st.selectbox('Credit History',("Unclear Debts", "No Unclear Debts"))
    
    result = ""
    
    if st.button("Predict"): 
        result = prediction(model_choice, Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success(f'Your loan is {result}')
     
if __name__ == '__main__': 
    main()
