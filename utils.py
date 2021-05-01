import re


def read_corpus(corpus_file):
    return [line.strip().split() for line in open(corpus_file, 'r')]


def read_corpus_conll(corpus_file, fs="\t"):
    featn = None
    sents = []
    words = []

    for line in open(corpus_file):
        line = line.strip()
        if len(line.strip()) > 0:
            feats = tuple(line.strip().split(fs))
            if not featn:
                featn = len(feats)
            elif featn != len(feats) and len(feats) != 0:
                raise ValueError("Unexpected number of column {} ({})".format(len(feats), featn))
            words.append(feats)
        else:
            if len(words) > 0:
                sents.append(words)
                words = []
    return sents


def parse_iob(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, None)


def get_chunks(corpus_file, fs="\t", otag="O"):
    sents = read_corpus_conll(corpus_file, fs=fs)
    return set([parse_iob(token[-1])[1] for sent in sents for token in sent if token[-1] != otag])


def get_column(corpus, column=-1):
    return [[word[column] for word in sent] for sent in corpus]


def compute_frequency_list(corpus):
    frequencies = {}
    for sent in corpus:
        for token in sent:
            frequencies[token] = frequencies.setdefault(token, 0) + 1
    return frequencies


def cutoff(corpus, tf_min=2):
    frequencies = compute_frequency_list(corpus)
    return sorted([token for token, frequency in frequencies.items() if frequency >= tf_min])


def make_w2t(isyms, osyms, out='w2t.tmp'):
    special = {'<epsilon>', '<s>', '</s>'}
    oov = '<UNK>'  # unknown symbol
    state = '0'  # wfst specification state
    fs = " "  # wfst specification column separator

    ist = sorted(list(set([line.strip().split("\t")[0] for line in open(isyms, 'r')]) - special))
    ost = sorted(list(set([line.strip().split("\t")[0] for line in open(osyms, 'r')]) - special))

    with open(out, 'w') as f:
        for i in range(len(ist)):
            for j in range(len(ost)):
                f.write(fs.join([state, state, ist[i], ost[j]]) + "\n")
        f.write(state + "\n")


# modified version to support fst-output
def read_fst4conll(fst_file, fs="\t", oov='<UNK>', otag='O', sep='+', split=False):
    sents = []  # list to hold words list sequences
    words = []  # list to hold feature tuples

    for line in open(fst_file):
        line = line.strip()
        if len(line.strip()) > 0:
            feats = tuple(line.strip().split(fs))
            # arc has minimum 3 columns, else final state
            if len(feats) >= 3:
                ist = feats[2]  # 3rd column (input)
                ost = feats[3]  # 4th column (output)
                # replace '<unk>' with 'O'
                ost = otag if ost == oov else ost
                # ignore for now
                ost = ost.split(sep)[1] if split and ost != otag else ost

                words.append((ist, ost))
            else:
                sents.append(words)
                words = []
        else:
            if len(words) > 0:
                sents.append(words)
                words = []
    return sents


# TODO: check why change ngram_words
# def make_w2t_mle(probs, out='w2t_mle.tmp'):
#     special = {'<epsilon>', '<s>', '</s>'}
#     oov = '<UNK>'  # unknown symbol
#     state = '0'  # wfst specification state
#     fs = " "  # wfst specification column separator
#     otag = 'O'
#     mcn = 3  # minimum column number
#
#     lines = [line.strip().split("\t") for line in open(probs, 'r')]
#
#     with open(out, 'w') as f:
#         for line in lines:
#             ngram = line[0]
#             ngram_words = ngram.split()  # by space
#             if len(ngram_words) == 2:
#                 if set(ngram_words).isdisjoint(set(special)):
#                     if ngram_words[1] in [otag, oov]:
#                         f.write(fs.join([state, state] + ngram_words + [line[1]]) + "\n")
#                     elif ngram_words[1].startswith("B-") or ngram_words[1].startswith("I-"):
#                         f.write(fs.join([state, state] + line) + "\n")
#         f.write(state + "\n")


def make_w2t_mle(probs, out='w2t_mle.tmp'):
    special = {'<epsilon>', '<s>', '</s>'}
    oov = '<UNK>'  # unknown symbol
    state = '0'  # wfst specification state
    fs = " "  # wfst specification column separator
    otag = 'O'
    mcn = 3  # minimum column number

    lines = [line.strip().split("\t") for line in open(probs, 'r')]

    with open(out, 'w') as f:
        for line in lines:
            ngram = line[0]
            ngram_words = ngram.split()  # by space
            if len(ngram_words) == 2:
                if set(ngram_words).isdisjoint(set(special)):
                    if ngram_words[0] in [otag, oov]:
                        f.write(fs.join([state, state] + ngram_words + [line[1]]) + "\n")
                    elif ngram_words[0].startswith("B-") or ngram_words[0].startswith("I-"):
                        f.write(fs.join([state, state] + line) + "\n")
        f.write(state + "\n")


def make_w2t_wt(isyms, sep='+', out='w2wt.tmp'):
    special = {'<epsilon>', '<s>', '</s>'}
    oov = '<UNK>'  # unknown symbol
    state = '0'  # wfst specification state
    fs = " "  # wfst specification column separator

    ist = sorted(list(set([line.strip().split("\t")[0] for line in open(isyms, 'r')]) - special))

    with open(out, 'w') as f:
        for e in ist:
            f.write(fs.join([state, state, e.split('+')[0], e]) + "\n")
        f.write(state + "\n")
