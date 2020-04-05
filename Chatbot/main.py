import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import random
import json
import pickle
import os
import time
import speech_recognition as sr
from gtts import gTTS
import playsound


with open("intents.json") as file:
    data_set = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, tags, train, output = pickle.load(f)

except:
    words = []
    tags = []
    pattern_words = []
    pattern_tags = []

    for intent in data_set["intents"]:
        for pattern in intent["questions"]:
            # Tokenize the words and store in 'words'
            pt = nltk.word_tokenize(pattern)
            words.extend(pt)
            pattern_words.append(pt)
            pattern_tags.append(intent["tag"])

        if intent["tag"] not in tags:
            tags.append(intent["tag"])

    # print(pattern_words)
    # print(words)

    """
    #pattern_words = [stemmer.stem(w.lower()) for w in pattern_words]
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    tags = sorted(tags)

    cv = CountVectorizer(max_features=1000)
    train = cv.fit_transform(words).toarray()

    print(train[0])
    """

    stemmer = LancasterStemmer()
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    tags = sorted(tags)

    train = []
    output = []

    output_empty = [0 for _ in range(len(tags))]

    for x, w in enumerate(pattern_words):
        bag_of_words = []

        wd = [stemmer.stem(wr.lower()) for wr in w]
        for wr in words:
            if wr in wd:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)

        row = output_empty[:]
        row[tags.index(pattern_tags[x])] = 1
        train.append(bag_of_words)
        output.append(row)

    train = np.array(train)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, tags, train, output), f)


def play_sound(text):
    text_speech = gTTS(text=text, lang="en")
    file = "voice.mp3"
    text_speech.save(file)
    playsound.playsound(file)


tf.reset_default_graph()

network = tflearn.input_data(shape=[None, len(train[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(output[0]), activation="softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)
"""
try:
    model.load("model.tflearn")
    
except:
"""
model.fit(train, output, n_epoch=500, batch_size=8, show_metric=True)
model.save("model.tflearn")


def input_words(sentence, word):
    bag = [0 for _ in range(len(word))]
    stm = LancasterStemmer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stm.stem(w.lower()) for w in sentence_words]

    for s in sentence_words:
        for i, j in enumerate(word):
            if j == s:
                bag[i] = 1

    return np.array(bag)


def start_chat():
    print("Welcome! Get ready to talk to Boss!")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "exit":
            break   

        results = model.predict([input_words(sentence, words)])
        tag = tags[np.argmax(results)]

        for tg in data_set["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                answer = random.choice(responses)
                print(answer)
                play_sound(answer)
                break


start_chat()

