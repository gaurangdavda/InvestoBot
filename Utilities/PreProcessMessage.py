import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.lancaster import LancasterStemmer

class PreProcessMessage():

    def __init__(self):
        self.stemmer = LancasterStemmer()

    def preprocess_word(self, word):
        # Remove punctuation
        word = word.strip('\'"?!,.():;')
        # Convert more than 2 letter repetitions to 2 letter
        # funnnnny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # Remove - & '
        word = re.sub(r'(-|\')', '', word)
        return word


    def is_valid_word(self, word):
        # Check if word begins with an alphabet
        return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


    def handle_emojis(self, message):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        message = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', message)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        message = re.sub(r'(:\s?d|:-d|x-?d|x-?d)', ' EMO_POS ', message)
        # Love -- <3, :*
        message = re.sub(r'(<3|:\*)', ' EMO_POS ', message)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        message = re.sub(r'(;-?\)|;-?d|\(-?;)', ' EMO_POS ', message)
        # Sad -- :-(, : (, :(, ):, )-:
        message = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', message)
        # Cry -- :,(, :'(, :"(
        message = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', message)
        return message

    def preprocess_message(self, message):
        processed_message = []
        # Convert to lower case
        message = message.lower()
        # Replaces URLs with the word URL
        message = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', message)
        # Replace 2+ dots with space
        message = re.sub(r'\.{2,}', ' ', message)
        # Strip space, " and ' from message
        message = message.strip(' "\'')
        # Replace emojis with either EMO_POS or EMO_NEG
        message = self.handle_emojis(message)
        # Replace multiple spaces with a single space
        message = re.sub(r'\s+', ' ', message)
        words = message.split()

        for word in words:
            word = self.preprocess_word(word)
            if self.is_valid_word(word):
                processed_message.append(word)
        words = ' '.join([str(item) for item in processed_message])
        return words

    def transformCleanedMessage(self, message):
        tfv=TfidfVectorizer(min_df=0, max_features=None, strip_accents='unicode',lowercase =True,
        analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1), sublinear_tf=True, stop_words = "english")
        transformedMessage=tfv.fit_transform(message)
        return transformedMessage

    def tokenizeCleanedMessage(self, message):
        wordlist = []
        wrds = nltk.word_tokenize(message)
        wordlist.extend(wrds)
        wordlist = [self.stemmer.stem(w.lower()) for w in wordlist if w != "?"]
        wordlist = sorted(list(set(wordlist)))
        return wordlist

    def bagOfWords(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
                
        return np.array(bag)