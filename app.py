import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np

from keras.models import load_model
import json
import random

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#from collections import Counter
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

emotion_counts = {
    "happy": 0,
    "sad": 0,
    "disgust": 0,
    "surprise": 0,
    "love": 0,
    "fear": 0,
    "angry": 0,
}

emotion_order_index = {}
emotion_analysis_counter = 0

def preprocess_for_emotion_analysis(text):
    # Convert text to lowercase
    lower_case = text.lower()
    # Remove punctuation
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokenized_words = word_tokenize(cleaned_text)
    # Remove stopwords excluding 'not' and similar negation terms
    custom_stopwords = set(stopwords.words('english')) - {'not', "no", "nor", "neither", "never","n't"}
    filtered_words = [word for word in tokenized_words if word not in custom_stopwords]
    return filtered_words

# Function to analyze emotion in the user's input message
def analyze_emotion(text):
    global emotion_analysis_counter
    # Preprocess the text to make it more simpler
    preprocessed_text = preprocess_for_emotion_analysis(text)
    lemma_words = []
    for word in preprocessed_text:
        word = lemmatizer.lemmatize(word)
        lemma_words.append(word)
    print("Lemma Words:", lemma_words)

    # Perform emotion analysis to check for word in emotions.txt
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word in lemma_words:
                emotion_list.append(emotion)
    # Determine the detected emotion
    detected_emotion = emotion_list[0] if emotion_list else "neutral"

    # Check for negations and reverse emotions if present
    negation_words = ["not", "no", "n't", "never", "don't"]
    negation_flag = any(word in lemma_words for word in negation_words)
    print("Negation Flag:", negation_flag)

    if negation_flag:
            # Handle negation for detected emotion
            if detected_emotion == "happy":
                detected_emotion = "sad"
            elif detected_emotion == "sad":
                detected_emotion = "happy"
            elif detected_emotion == "disgust":
                detected_emotion = "neutral"
            elif detected_emotion == "surprised":
                detected_emotion = "neutral"
            elif detected_emotion == "neutral":
                detected_emotion = "neutral"
            elif detected_emotion == "love":
                detected_emotion = "disgust"
            elif detected_emotion == "fear":
                detected_emotion = "love"
            elif detected_emotion == "angry":
                detected_emotion = "happy"
            elif detected_emotion == "love":
                detected_emotion = "angry"
    # Update emotion counts and order index
    if detected_emotion in emotion_counts:
        emotion_counts[detected_emotion] += 1
        emotion_analysis_counter += 1
        emotion_order_index[detected_emotion] = emotion_analysis_counter
    return detected_emotion

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request, jsonify
from urllib.parse import unquote
import requests

app = Flask(__name__)
app.static_folder = 'static'

# Last.fm API parameters
api_key = '454107d2d830a87aa7b18b82c61f8397'
api_url = 'http://ws.audioscrobbler.com/2.0/'

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    x= unquote(userText)
    # Analyze emotion from the user's message
    detected_emotion = analyze_emotion(x)
    # Increment the count for the detected emotion
    #if detected_emotion in emotion_counts:
    #    emotion_counts[detected_emotion] += 1
    print("\n",x,"\n")
    print("Detected Emotion:", detected_emotion)
    response= chatbot_response(userText)
    return jsonify(response)

@app.route("/get_emotion_counts", methods=["GET"])
def get_emotion_counts():
    global emotion_counts, emotion_order_index

    # Find the maximum count among all emotions
    max_count = max(emotion_counts.values())
    # Get all emotions that have the maximum count
    predominant_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]
     # Choose the emotion with the highest order index if there are ties in counts

    if len(predominant_emotions) > 1:
        predominant_emotions = sorted(predominant_emotions, key=lambda e: emotion_order_index[e], reverse=True)
    latest_emotion = predominant_emotions[0]  # Choose the first emotion with the highest order index

    print("Current Emotion Counts:", emotion_counts)
    print("Predominant Emotion(s):", predominant_emotions)
    # Search for recommended tracks based on the predominant emotion
    recommended_tracks = search_tracks_by_emotion(latest_emotion)

    return jsonify({"emotion": latest_emotion, "tracks": recommended_tracks})

# Function to search for tracks by tag (emotion)
def search_tracks_by_emotion(emotion):
    params = {
        'method': 'tag.gettoptracks',
        'tag': emotion,
        'api_key': api_key,
        'format': 'json',
        'limit': 100  # Limiting to 100 tracks for simplicity
    }

    response = requests.get(api_url, params=params)
    data = response.json()

    # Check if the request was successful
    if 'tracks' in data:
        tracks = data['tracks']['track']
        track_list = []
        for track in tracks:
            track_dict = {
                'name': track['name'],
                'artist': track['artist']['name'],
                'url': track['url']
            }
            track_list.append(track_dict)
        
        # Select 7 random items from track_list
        random_tracks = random.sample(track_list, 7)
        
        return random_tracks
    else:
        return None

@app.route("/search_tracks", methods=["GET"])
def search_tracks():
    emotion = request.args.get('emotion')
    if emotion:
        tracks = search_tracks_by_emotion(emotion)
        #sent back from web server to client application using jsonify
        return jsonify(tracks)
    else:
        return jsonify([])  # Return empty list if no emotion is specified

if __name__ == "__main__":
    app.run() #to run the flask app

