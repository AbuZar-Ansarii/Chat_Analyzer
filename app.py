import streamlit as st
from collections import Counter
import pandas as pd
import processor,helper
import matplotlib.pyplot as plt

st.sidebar.title(" Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = processor.process(data)
    df["Name"] = df["Name"].apply(processor.clear_df)
    st.dataframe(df)
    user_list = df["Name"].unique().tolist()
    user_list.insert(0,"Overall")
    selected_user = st.sidebar.selectbox("select user",user_list)
    if st.sidebar.button("Show Analysis"):
        num_messages,t_words = helper.fetch_stats(selected_user,df)
        st.title("Statistics")
        col1,col2 = st.columns(2)

        with col1:
            st.header("Total Messages")
            st.header(num_messages)
        with col2:
            st.header("Total Words")
            st.header(t_words)
        # User chat percentage
        st.header("User Chat Percentage")
        fig, ax = plt.subplots()
        ax.pie((df["Name"].value_counts().head() / df.shape[0]), labels=df["Name"].value_counts().head(),autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

        # finding the busy member of group chat

        if selected_user == "Overall":
            st.title("Busy Members")
            col1,col2 = st.columns(2)
            x,dff = helper.busy_user(df)
            fig,ax = plt.subplots()

            with col1:
                st.header("Most Busy Member")
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation="vertical")
                st.pyplot(fig)
            with col2:
                st.header("Top Busy Member")
                st.dataframe(dff)

        # word cloud
        st.header("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
        # Common words
        words = []
        for i in df["Chat"]:
            words.extend(i.split())
        cleaned_list = helper.clean_text(words)
        wd_df = pd.DataFrame(Counter(words).most_common(20), columns=["Words", "Count"])
        st.header("Most Common Words Bar")
        fig,ax = plt.subplots()
        ax.barh(wd_df["Words"],wd_df["Count"])
        plt.xticks(rotation="vertical")
        st.pyplot(fig)





