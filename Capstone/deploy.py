import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
from plotly import graph_objs as go
import plotly.express as px

# Judul utama aplikasi
st.set_page_config(page_title="Sentiment Analysis IKN Tweet", layout="wide")
st.title("\U0001F5E3\ufe0f Sentiment Analysis for IKN Tweets (@tanyarlfes)")

# Membaca data
all_data = pd.read_csv('ikn-tanyarlfes.csv')
clean_data = pd.read_csv('ikn-clean-sentiment.csv')

# Garis pemisah visual
st.markdown("""
    <hr style="border: 2px solid #4CAF50; border-radius: 5px;">
""", unsafe_allow_html=True)

# Visualisasi Sentimen Keseluruhan
st.subheader("\U0001F4CA Sentiment Overview")
freq = pd.Series(' '.join(clean_data['text_clean']).split()).value_counts()
head_freq = freq.head(20)
fig1 = px.bar(
    head_freq, x=head_freq.index, y=head_freq.values, 
    labels={'x': 'Words', 'y': 'Frequency'},
    title="Top 20 Words by Frequency"
)
fig1.update_layout(
    title_font_size=20, xaxis_tickangle=-45, template="plotly_dark"
)
st.plotly_chart(fig1, use_container_width=True)

# Distribusi Sentimen
st.markdown("""### Distribution of Sentiment Labels""")
temp = clean_data.groupby('Sentiment').count()['text_clean'].reset_index().sort_values(by='text_clean', ascending=False)
col1, col2 = st.columns(2)
with col1:
    fig2 = px.bar(temp, x='Sentiment', y='text_clean', color='Sentiment',
                  title="Bar Chart of Sentiment Distribution",
                  labels={'text_clean': 'Count'})
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = px.pie(temp, values='text_clean', names='Sentiment', hole=0.4,
                  title="Sentiment Proportion")
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig3, use_container_width=True)

# Garis pemisah visual
st.markdown("""
    <hr style="border: 2px solid #4CAF50; border-radius: 5px;">
""", unsafe_allow_html=True)

# Visualisasi Berdasarkan Sentimen
for sentiment_label, emoji in zip(["Positif", "Negatif", "Netral"], ["\U0001F600", "\U0001F641", "\U0001F610"]):
    st.subheader(f"{emoji} {sentiment_label} Sentiment Visualization")
    df_sent = clean_data[clean_data['Sentiment'] == sentiment_label]
    word_freq = pd.Series(' '.join(df_sent['text_clean']).split()).value_counts()
    top_words = word_freq.head(10)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(top_words, y=top_words.index, x=top_words.values, orientation='h',
                     labels={'x': 'Frequency', 'y': 'Words'},
                     title=f"Top 10 Words in {sentiment_label} Sentiment")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        wordcloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(' '.join(df_sent['text_clean']))
        fig_cloud, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis('off')
        ax.set_title(f"Word Cloud for {sentiment_label} Sentiment", fontsize=16)
        st.pyplot(fig_cloud)

# Garis pemisah visual
st.markdown("""
    <hr style="border: 2px solid #4CAF50; border-radius: 5px;">
""", unsafe_allow_html=True)

# Prediksi Sentimen
st.header("\U0001F50E\ufe0f Sentiment Prediction System")
new_text = st.text_input("Masukkan teks untuk prediksi sentimen:")
model = joblib.load('naive_bayes_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

if st.button("Predict Sentiment"):
    if not new_text.strip():
        st.warning("Please enter text to predict.")
    else:
        transformed_text = tfidf_vectorizer.transform([new_text])
        prediction = model.predict(transformed_text)
        st.success(f"Predicted Sentiment: {prediction[0]}")

# Prediksi Sentimen dari File
st.header("\U0001F4C2 Predict Sentiment from File")
uploaded_file = st.file_uploader("Upload a text file (.txt):", type="txt")

if st.button("Predict Sentiment from File"):
    if not uploaded_file:
        st.warning("Please upload a file.")
    else:
        file_content = uploaded_file.read().decode("utf-8")
        sentences = file_content.splitlines()

        if not sentences or all(s.strip() == "" for s in sentences):
            st.error("The file is empty or does not contain valid text.")
        else:
            results = []
            for sentence in sentences:
                if sentence.strip():
                    transformed_text = tfidf_vectorizer.transform([sentence])
                    prediction = model.predict(transformed_text)[0]
                    results.append((sentence, prediction))

            st.subheader("Prediction Results:")
            for sentence, sentiment in results:
                st.write(f"**Text**: {sentence}")
                st.write(f"**Predicted Sentiment**: {sentiment}")
                st.markdown("---")
