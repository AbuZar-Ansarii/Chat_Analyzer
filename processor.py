import pandas as pd
import re
import string

def process(data):
    pattern = r"\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}\u202f[ap]m - [^:]+: " # added the delimiter pattern to the regex
    chat = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    date = []
    for i in dates:
        date.append(i.split(',')[0])
    name = []
    for i in dates:
        # Check if the delimiter is present before splitting
        if ' - ' in i:
            name.append(i.split(' - ')[1].split(':')[0]) # Splitting on ':' to remove trailing colon and spaces
        else:
            # Handle cases where the delimiter is missing
            # for example, append a default value like 'Unknown'
            name.append('Unknown')
    df = pd.DataFrame({'Name': name, 'Date': date, 'Chat': chat})
    df['Date'] = pd.to_datetime(df['Date'])
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Year"] = df["Date"].dt.year

    return df
def clear_df(text):
        return text.translate(str.maketrans('', '', string.punctuation))

