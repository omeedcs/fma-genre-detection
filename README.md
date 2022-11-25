# Genre Detection and Model Evaluation with FMA using Traditional Machine Learning / Modern Approaches for Audio Classification

## Contributors: 
Kunal Mody, Omeed Tehrani, Alex Chandler, Felipe Paz

## Official Title: 
Genre Detection and Model Evaluation with FMA using Traditional Machine Learning / Modern Approaches for Audio Classification

## General Overview: 
This project compares traditional machine learning techniques to more modern approaches for audio genre classification. 

## Getting Started:
This project has been transfered to a a Jupyter Notebook, but traditionally, the following steps would be used to create a virtual environment and install the necessary modules

'''virtualenv music_genre_env source music_genre_env/bin/activate pip3 install -r requirements.txt'''

This jupyter notebook requires the mounting of a google drive. Code for that is in the notebook.

### Enabling GPU
Runtime -> Change runtime type -> Hardware accelerator -> GPU

## Introduction:

1. What is the machine learning problem we are trying to solve? <br>
Our project goal is to classify MP3 audio data of 30 second snippets of songs into 8 genres.
Audio classification is a challenging problem in machine learning, especially with traditional machine learning methods such as regression, and decision trees. Any success without utilizing recent improvements in the field requires finding meaningful signals or features from audio pre-processing methods.

2. Why does this problem matter? <br>
Audio services such as Spotify, Apple Music, or Shazam, would desire being able to classify music into genres without having access to a database that labels each incoming uploaded audio clip to a certain genre. Additionally, anomaly detection could be performed on existing data that maps audio files to genre. As discussed later in the notebook, existing audio genre data is not fully reliable and full of classified labels that your average listener would disagree with. Attempting this problem also serves a way to learn and sharpen one's skills in processing audio data. Throwing the resampled data into a traditional machine learning method will barely outperform a random guess factoring in the genre probabilities without significant data pre-processing. In effect, this project forces us to learn of methods to extract useful information that could then be fed into a method that does not utilize convolution.

3. What could the results of our predictive model be used for?<br>
Our predictive model, or any current existing, publicically available, model, is not accurate enough to reliably place audio/label data into genres. However, there will be many useful applications once a model is capable of achieving high precision and recall. As discussed in the previous answer, audio services could use the classifier to quality check the labled data by performing anomoly detection. A company could classify newly submitted user-generated audio data into genres, potentially filtering out undesired audio data. 

4. Why would we want to be able to predict genres using this data?<br>

There are a range of applications for this project. 
1. Let's say an artist tries to change the label of their songs to try and gain more streams or attention on a platform like Spotify. By being able to predict the genre of a song using the mp3 data, we can avoid that by creating a robust filtration system. 
2. Instead of using purely search feedback to give a user recommendations on songs, we can use the actual sound of the song to predict future genres and songs similar to a particular genre that the user would enjoy.
3. This methodology attempts to corroborate the idea that good data processing can lead to higher accuracy and scoring, despite the model not being designed for this complex data type. We also want to show, that newer machine learning architectures can be simpler and result in better accuracy on complex data types.

## Description of Dataset

We use the FMA dataset. This dataset is known as the “Free Music Archive”. It is a database of 106,574 songs. Features included in the dataset are title, artist, genres, tags and play counts, for all 106,574 tracks. The MP3-encoded audio data is available in four sizes, ranging from a sample of 8,000 tracks to the full untrimmed unbalanced 106,574 tracks (800+ gigabytes).

There are 106,574 records from 16,341 artists and 14,854 albums. It is important to note that we have access to genre-balanced subsets. 

To minimize memory usage, we use FMA-small, a subset of the dataset which contains 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB). It is important to note that this dataset has several issues, including mislabled audio genres and poor data organization.

We have over 50 features for each song, although we will only initially be using the mp3 and genre features. Some of the features are frequently missing, such as ‘engineer’ or ‘comments’. 

List of features from original dataset: ['trackid', 'album', 'comments', 'date', 'created', 'date', 'released', 'engineer', 'favorites', 'id', 'information', 'listens', 'producer', 'tags', 'title', 'tracks', 'type', 'active', 'year', 'begin', 'active', 'year', 'end', 'associated', 'labels', 'bio', 'comments', 'date', 'created', 'favorites', 'id', 'latitude', 'location', 'longitude', 'members', 'name', 'related', 'projects', 'tags', 'website', 'wikipedia', 'page', 'split', 'subset', 'bit', 'rate', 'comments', 'composer', 'date', 'created', 'date', 'recorded', 'duration', 'favorites', 'genre', 'top', 'genres', 'genres', 'all', 'information', 'interest', 'language', 'code', 'license', 'listens', 'lyricist', 'number', 'publisher', 'tags', 'title']

The only feature that we use from the original data is the genre and the mp3 audio.
This dataset contains the labels we are trying to predict (the genres of each song), hinting at supervised learning approaches.


The publication for this dataset is available at: https://arxiv.org/abs/1612.01840

This dataset is available at: https://github.com/mdeff/fma.

## Contact Us

Please reach out to omeed@cs.utexas.edu, alex.chandler@utexas.edu, felipeps2020@gmail.com, kunalmody1@gmail.com for any questions you might have.

## Evaluation
We evaluate the genre classification using both traditional error metrics including recall, precision, accuracy. One challenge with evaluating accuracy is that a song can be multiple genres. A song, for example, can be both acoustic and pop.
