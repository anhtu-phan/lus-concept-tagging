import re
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import nltk
from gensim.utils import simple_preprocess

np.random.seed(2018)
nltk.download("wordnet")
stemmer = SnowballStemmer('english')


def represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def remove_stop_words(input_utter_file, input_conll_file, output_utter_file, output_conll_file, training=True):
    stop_words = {}
    with open("./dataset/english.stop.txt") as f:
        for line in f:
            word = line.strip()
            if word not in stop_words:
                stop_words[word] = 1
    utter_sentences = [line.strip().split() for line in open(input_utter_file)]

    it_conl_sen = 0
    with open(input_conll_file, 'r') as f:
        with open(output_conll_file, 'w') as f_conll_out:
            with open(output_utter_file, 'w') as f_utter_out:
                removed_words_index = []
                word_pos = 0
                for line in f:
                    line = line.strip()
                    if len(line.strip()) > 0:
                        line = line.strip().split("\t")
                        word = line[0]
                        tag = line[1].strip()
                        if training:
                            if word in stop_words and tag == "O":
                                removed_words_index.append(word_pos)
                            else:
                                f_conll_out.write("\t".join(line) + "\n")
                        else:
                            if word in stop_words:
                                removed_words_index.append(word_pos)
                            else:
                                f_conll_out.write("\t".join(line) + "\n")
                        word_pos += 1
                    else:
                        f_conll_out.write("\n")
                        removed_utter_sent = []
                        for i, word in enumerate(utter_sentences[it_conl_sen]):
                            if i not in removed_words_index:
                                removed_utter_sent.append(word)
                        f_utter_out.write(" ".join(removed_utter_sent) + "\n")
                        removed_words_index = []
                        word_pos = 0
                        it_conl_sen += 1


def norm_data_input(input_file, output_file, file_type="utter"):
    with open(output_file, 'w') as f_out:
        # question_words = ["which", "what", "whose", "who", "whom", "where", "when", "how", "why"]
        with open(input_file, 'r') as f_in:
            for line in f_in:
                if file_type == 'utter':
                    words = line.strip().split()
                    for i, w in enumerate(words):
                        if represent_int(w):
                            words[i] = "<num>"
                        else:
                            words[i] = lemmatize_stemming(w)
                else:
                    words = line.strip().split("\t")
                    if represent_int(words[0]):
                        words[0] = "<num>"
                    else:
                        words[0] = lemmatize_stemming(words[0])

                # for w in question_words:
                #     if w in words:
                #         words[words.index(w)] = "<question_word>"
                if file_type == 'utter':
                    f_out.write(" ".join(words) + "\n")
                else:
                    f_out.write("\t".join(words) + "\n")


norm_data_input("dataset/NL2SparQL4NLU.train_no_stop_word.utterances.txt",
                "dataset/NL2SparQL4NLU.train_norm_all_words_no_stop_word.utterances.txt")
norm_data_input("dataset/NL2SparQL4NLU.test_no_stop_word.utterances.txt",
                "dataset/NL2SparQL4NLU.test_norm_all_words_no_stop_word.utterances.txt")
norm_data_input("dataset/NL2SparQL4NLU.train_no_stop_word.conll.txt",
                "dataset/NL2SparQL4NLU.train_norm_all_words_no_stop_word.conll.txt",
                file_type='conll')
norm_data_input("dataset/NL2SparQL4NLU.test_no_stop_word.conll.txt",
                "dataset/NL2SparQL4NLU.test_norm_all_words_no_stop_word.conll.txt",
                file_type='conll')
# remove_stop_words("dataset/NL2SparQL4NLU.train_norm_all_words.utterances.txt",
#                   "dataset/NL2SparQL4NLU.train_norm_all_words.conll.txt",
#                   "dataset/NL2SparQL4NLU.train_norm_all_words_no_stop_word.utterances.txt",
#                   "dataset/NL2SparQL4NLU.train_norm_all_words_no_stop_word.conll.txt")
# remove_stop_words("dataset/NL2SparQL4NLU.test_norm_all_words.utterances.txt",
#                   "dataset/NL2SparQL4NLU.test_norm_all_words.conll.txt",
#                   "dataset/NL2SparQL4NLU.test_norm_all_words_no_stop_word.utterances.txt",
#                   "dataset/NL2SparQL4NLU.test_norm_all_words_no_stop_word.conll.txt", training=False)
