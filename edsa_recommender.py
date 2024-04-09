"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import pandas as pd                                                  
import numpy as np                                                    
import matplotlib.pyplot as plt                                                                                    
import seaborn as sns                                                                                                       
import scipy as sp                                                     
import plotly.express as px                                           
from PIL import Image
from wordcloud import WordCloud, STOPWORDS                            
sns.set()                                                             

# Libraries for data preparation
from datetime import datetime
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet

# Libraries for featurization and similarity computation
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
#from streamlit_option_menu import option_menu                                            


# Custom Libraries
import os
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "Company Information","Why choose spark?","Conclusion and future recommendations","App Feedback"]
    

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image("https://th.bing.com/th/id/OIP.QKitTPB59SiGnedioysPWwHaFl?rs=1&pid=ImgDetMain")

        
        
        if st.button("Introduction"):
            st.image("https://media3.giphy.com/media/3o7qDV67vk5vYdoeVG/giphy.gif", width= 500)
        
            st.write("-Background information on recommender systems.")
            st.write("-Problem statement.")
            st.write("-The aim of this project")
            st.write("-Information about the target market(Who are we seling the app to?)")
            st.write("-Solution to the problem statement")

        if st.button("Introduction of the app"):
            st.markdown("Welcome to Spark.")
            st.image("resources\imgs\Streamlit_App.jpeg", width= 300)
            st.markdown("Welcome to Spark, your personalized recommendation companion! ✨ Get ready to ignite your interests and discover your next favorite content with Spark by your side. Let's light up your streaming journey together!"
)
        
        if st.button('EDA'):
            #ratings_df = pd.read_csv('ratings.csv')
            st.title("Exploratory Data Analysis")
            st.text('')
            st.subheader("Data Visualization")
            st.image("resources\imgs\movie_ratings.png")
            st.image("resources\imgs\Distribution of movie genres.png")
            st.markdown("By visualizing the distribution of movie genres, streaming platforms like Netflix can gain insights into the types of content that are most popular among their users. This understanding can inform content acquisition strategies, content production decisions, and recommendation algorithms to better match user preferences. As seen on the bargraph that the most popular Genre is Drama and this is sigficant because Netflix would know which genre to realse frequently in order to keep up with the demnads of viewers .")

        if st.button('MODEL SELECTION'):
            st.image('resources\imgs\model selection.png')

       
    if page_selection == "Company Information":
        '# ABOUT THE COMPANY'
        st.image("resources\imgs\company_logo.png", width=400)
        st.markdown("CODENEST! Established in 2023, we're dedicated to transforming the entertainment landscape. Our vision is to enhance the movie-watching experience for all. By harnessing data science, we provide personalized recommendations and curated content. Our mission is to delight movie lovers worldwide with innovative solutions and immersive experiences. Join us as we shape the future of entertainment together!")
        '# MEET THE TEAM'
        st.image("resources\imgs\FM1  (3).jpg")

        st.markdown('Contact Us')
        st.image('resources\imgs\Contact us.jpg')

    

    if page_selection == 'Why choose spark?':
            st.image("https://thumbs.dreamstime.com/z/woman-shrugging-hands-expressing-confusion-not-having-any-clue-european-your-question-my-problem-studio-shot-143954121.jpg", width=500)
            st.markdown('-User-Friendly Interface,therefore enhances new user experience')
            st.markdown('-Time saving-by eliminating the need to manually search for content,With personalized recommendations, users can quickly find movies.')
            st.markdown('-Free trial or freemium model- to encourage users to try out our app risk-free.')
            st.markdown('-Can be continuously improved and updated')
            st.markdown('-Easy discovery of New Content')

        
    if page_selection == 'Conclusion and future recommendations':
        st.image("https://www.mkgifs.com/wp-content/uploads/2023/08/Clipart-Shooting-Star-GIF.gif", width= 300)
        st.markdown("Spark is more than just a recommendation app; its your personalized guide to an enriched entertainment experience. With its intuitive interface and cutting-edge recommendation algorithms, Spark empowers users to discover content that resonates with their unique preferences and interests. From movies and TV shows to music and books, Spark opens the door to a world of endless entertainment possibilities.")
        st.header('-Q&A')  
        
        st.image('https://i.pinimg.com/originals/62/0f/39/620f39ce8c3d0eeb9ae1241f7b78f704.gif')

    if page_selection == "App Feedback":
        st.title("Please Rate this app!")
        if st.button("👍 Like"):
            st.write("Thank you for your feedback!")
        if st.button("👎 Dislike"):
			# Handle thumbs down action (e.g., decrement a counter)
            st.write("We appreciate your feedback!")
        
        


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
