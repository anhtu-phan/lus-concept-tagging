from utils import parse_iob


class DataAnalysis:
    def __init__(self, utter_file_path, conll_file_path):
        self.utter_file_path = utter_file_path
        self.conll_file_path = conll_file_path

    def count_word(self):
        num_word = 0
        with open(self.utter_file_path, 'r') as f:
            for line in f:
                num_word += len(line.strip().split())
        return num_word

    def count_sentence(self):
        return len([line for line in open(self.utter_file_path, 'r')])

    def count_lexicon(self):
        vocab = set()
        with open(self.utter_file_path, 'r') as f:
            for line in f:
                for word in line.strip().split():
                    vocab.add(word)
        return len(vocab)

    def get_vocab(self):
        vocab = {}
        with open(self.utter_file_path, 'r') as f:
            for line in f:
                for word in line.strip().split():
                    word = word.strip()
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        return vocab

    def get_tag_vocab(self, fs="\t", iob=True):
        if self.conll_file_path is None:
            raise FileNotFoundError("Can't find conll file in {} ".format(self.conll_file_path))
        vocab = {}
        with open(self.conll_file_path, 'r') as f:
            for line in f:
                line = line.strip().split(fs)
                if len(line) == 2:
                    tag = line[1].strip()
                    if not iob:
                        tag = parse_iob(tag)[1]
                        if tag is None:
                            tag = line[1].strip()
                    if tag in vocab:
                        vocab[tag] += 1
                    else:
                        vocab[tag] = 1
        return vocab

    def nbest(self, d, n=1, rev=True):
        """
        get n max values from a dict
        :param d: input dict (values are numbers, keys are stings)
        :param n: number of values to get (int)
        :return: dict of top n key-value pairs
        """
        return dict(sorted(d.items(), key=lambda item: item[1], reverse=rev)[:n])


d = DataAnalysis("./dataset/NL2SparQL4NLU.train_norm_all_words_no_stop_word.utterances.txt", "./dataset/NL2SparQL4NLU.train_norm_all_words_no_stop_word.conll.txt")
d2 = DataAnalysis("./dataset/NL2SparQL4NLU.train.utterances.txt", "./dataset/NL2SparQL4NLU.train.conll.txt")
print(d2.count_lexicon())
print(d2.nbest(d2.get_vocab(), 10, rev=True))
print(d.nbest(d.get_vocab(), 10, rev=True))
print(d2.nbest(d2.get_tag_vocab(), 10, rev=True))
print(d.nbest(d.get_tag_vocab(), 10, rev=True))
tags = d2.get_tag_vocab(iob=False)
print(d2.nbest(tags, len(tags), rev=True))
print(d.nbest(d.get_tag_vocab(iob=False), 10, rev=True))


# t = DataAnalysis("./dataset/NL2SparQL4NLU.test_no_stop_word.utterances.txt", "./dataset/NL2SparQL4NLU.test_no_stop_word.conll.txt")
# t2 = DataAnalysis("./dataset/NL2SparQL4NLU.test.utterances.txt", "./dataset/NL2SparQL4NLU.test.conll.txt")
# print(t2.count_lexicon())
# print(t2.nbest(t2.get_vocab(), 10, rev=True))
# print(t.nbest(t.get_vocab(), 10, rev=True))
# # print(t2.nbest(t2.get_tag_vocab(), 10, rev=True))
# # print(t.nbest(t.get_tag_vocab(), 10, rev=True))
# tags = t2.get_tag_vocab(iob=False)
# print(t2.nbest(tags, len(tags), rev=True))
# print(t.nbest(t.get_tag_vocab(iob=False), 10, rev=True))
