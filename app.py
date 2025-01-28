import streamlit as st
import pandas as pd
import pickle
import re

# Load models and data
nadi_model = pickle.load(open("nadi_model.pkl", "rb"))
admin_keys = pickle.load(open("admin_key.pkl", "rb"))
admin_values = pickle.load(open("admin_values.pkl", "rb"))

# Strings to remove
string_to_remove = "Start nPULSE001"
r_text = "Start"

# Streamlit UI
st.title("NADI PULSE DISEASE DETECTOR")

patient_age = st.number_input("Enter Patient Age", min_value=0, max_value=120)
nadi_patient_data = st.file_uploader("Upload patient data text file", type="txt")

# Read uploaded file
if nadi_patient_data is not None:
    nadi_text = nadi_patient_data.getvalue().decode("utf-8")  # Read text content
else:
    nadi_text = ""  # Default empty string to avoid errors

# Function to clean text
def clear_text(text):
    return text[:-25] if len(text) > 25 else text  # Ensure safe slicing

def remove_string_from_content(text, string_to_remove):
    return re.sub(string_to_remove, "", text).strip()

def remove_start_from_content(text, r_text):
    return re.sub(r_text, "", text).strip()

# Process text data
cleaned_text = clear_text(nadi_text)
cleaned_text = remove_string_from_content(cleaned_text, string_to_remove)
cleaned_text = remove_start_from_content(cleaned_text, r_text)

# Create DataFrame
patient_df = pd.DataFrame({"Patient Age": [patient_age], "nadi_data": [cleaned_text]})

# Function to process Nadi data
def process_nadi_data(column):
    def safe_convert_to_int(row):
        return [int(value) for value in row.split(",") if value.strip().isdigit()]

    processed_data = column.apply(lambda x: [safe_convert_to_int(row) for row in x.strip().split("\n") if row.strip()] if isinstance(x, str) else [])
    return processed_data

# Apply processing
patient_df["processed_nadi_data"] = process_nadi_data(patient_df["nadi_data"])
patient_df["sum_of_nadi_data"] = patient_df["processed_nadi_data"].apply(lambda x: sum(sum(sublist) for sublist in x))

input_data = [[patient_df["Patient Age"][0],patient_df["sum_of_nadi_data"][0]]]

pred_df = pd.DataFrame(input_data, columns=['Patient Age' ,'sum_of_nadi_data'])

def find_disease(pred_output):
    index = admin_keys.index(pred_output)
    get_desiese = admin_values[index]
    return get_desiese

# Display result
if st.button("Predict Disease"):
    st.dataframe(patient_df[["Patient Age", "sum_of_nadi_data"]])
    pred_output = nadi_model.predict(pred_df)[0]
    data_index = admin_keys.index(pred_output)
    st.header(f"Model's Prediction - {find_disease(pred_output)}")
