def remove_stop_words(input_utter_file, input_conll_file, output_utter_file, output_conll_file):
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
                removed_words = {}
                for line in f:
                    line = line.strip()
                    if len(line.strip()) > 0:
                        line = line.strip().split("\t")
                        word = line[0]
                        if word not in stop_words:
                            f_conll_out.write("\t".join(line)+"\n")
                        else:
                            if word not in removed_words:
                                removed_words[word] = 1
                    else:
                        f_conll_out.write("\n")
                        removed_utter_sent = []
                        for word in utter_sentences[it_conl_sen]:
                            if word not in removed_words:
                                removed_utter_sent.append(word)
                        f_utter_out.write(" ".join(removed_utter_sent)+"\n")
                        removed_words = {}
                        it_conl_sen += 1


remove_stop_words("dataset/NL2SparQL4NLU.train.utterances.txt", "dataset/NL2SparQL4NLU.train.conll.txt", "dataset/NL2SparQL4NLU.train_no_stop_word.utterances.txt", "dataset/NL2SparQL4NLU.train_no_stop_word.conll.txt")