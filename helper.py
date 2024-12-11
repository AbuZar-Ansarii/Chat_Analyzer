from wordcloud import WordCloud
import string
from nltk.corpus import stopwords
def fetch_stats(selected_user, df):
    # fetch messages
    if selected_user == "Overall":
        num_messages = df.shape[0]
        # fetch words
        words = []
        for i in df["Chat"]:
            words.extend(i.split())
            t_words = len(words)
        return num_messages, t_words
    else:
        new_df = df[df["Name"] == selected_user]
        num_messages = new_df.shape[0]
        words = []
        for i in new_df["Chat"]:
            words.extend(i.split())
            t_words = len(words)
        return num_messages, t_words


def busy_user(df):
    x = df["Name"].value_counts().head()
    dff = round((df["Name"].value_counts().head(10) / df.shape[0]) * 100, 2).reset_index().rename(
        columns={"index": "Name", "count": "Percentage"})
    return x,dff

def create_wordcloud(selected_user,df):
    if selected_user != "Overall":
        df = df[df["Name"] == selected_user]

    wc = WordCloud(width=500,height=500,max_font_size=60,background_color='white')
    df_wc = wc.generate(df["Chat"].str.cat(sep=" "))
    return df_wc


def clean_text(input_list):
    # Load English stopwords
    stop_words = set(stopwords.words('english'))

    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)

    cleaned_list = []
    for sentence in input_list:
        # Remove punctuation
        no_punctuation = sentence.translate(translator)
        # Split sentence into words, remove stopwords, and join back
        cleaned_sentence = ' '.join(
            word for word in no_punctuation.split() if word.lower() not in stop_words
        )
        cleaned_list.append(cleaned_sentence)

    return cleaned_list
