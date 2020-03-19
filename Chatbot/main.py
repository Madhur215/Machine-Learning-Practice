import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import random
import json

with open("intents.json") as file:
    data_set = json.load(file)

