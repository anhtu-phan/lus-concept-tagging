from conll import evaluate
import pandas as pd
import os
from utils import *
import argparse


class WfsmMle:
    def __init__(self, alg_type="mle", data_type="", ngram_order=1, back_off='false', smooth_method='witten_bell'):
        self.ALG_TYPE = alg_type
        self.DATA_TYPE = data_type
        self.NGRAM_ORDER = ngram_order
        self.BACK_OFF = back_off
        self.SMOOTH_METHOD = smooth_method

    def prepare_input(self):
        os.system(f'cp ./dataset/NL2SparQL4NLU.train{self.DATA_TYPE}.utterances.txt trn.txt')
        os.system(f'cp ./dataset/NL2SparQL4NLU.test{self.DATA_TYPE}.utterances.txt tst.txt')
        os.system(f'cp ./dataset/NL2SparQL4NLU.train{self.DATA_TYPE}.conll.txt trn.conll')
        os.system(f'cp ./dataset/NL2SparQL4NLU.test{self.DATA_TYPE}.conll.txt tst.conll')

        trn_data = read_corpus('trn.txt')
        trn_lex = cutoff(trn_data)

        with open('isyms.trn.txt', 'w') as f:
            f.write("\n".join(trn_lex) + "\n")

        os.system('ngramsymbols isyms.trn.txt isyms.txt')
        os.system("farcompilestrings --symbols=isyms.txt --keep_symbols --unknown_symbol='<UNK>' trn.txt trn.far")
        os.system("farcompilestrings --symbols=isyms.txt --keep_symbols --unknown_symbol='<UNK>' tst.txt tst.far")

    def generate_output(self):
        types = get_chunks('trn.conll')
        with open('osyms.u.lst.txt', 'w') as f:
            # let's add 'O'
            f.write("O" + "\n")
            for c in sorted(list(types)):
                # prefix each type with segmentation information
                f.write("B-" + c + "\n")
                f.write("I-" + c + "\n")
        os.system('ngramsymbols osyms.u.lst.txt osyms.txt')

    def create_test_set(self):
        os.system('mkdir -p wdir')
        os.system('farextract --filename_prefix="wdir/" tst.far')

    def lm_create_training_data(self):
        trn = read_corpus_conll('trn.conll')
        tags = get_column(trn, column=-1)
        # write data
        with open('trn.t.txt', 'w') as f:
            for s in tags:
                f.write(" ".join(s) + "\n")
        tlex = cutoff(tags)
        with open('osyms.t.lst.txt', 'w') as f:
            f.write("\n".join(tlex) + "\n")
        os.system("ngramsymbols osyms.t.lst.txt osyms.t.txt")
        # os.system("ngramsymbols osyms.u.lst.txt osyms.t.txt")
        os.system("farcompilestrings --symbols=osyms.t.txt --keep_symbols --unknown_symbol='<UNK>' trn.t.txt trn.t.far")

    def lm_train_unigram(self):
        os.system(f"ngramcount --order={self.NGRAM_ORDER} trn.t.far trn.t1.cnt")
        os.system(f"ngrammake --method={self.SMOOTH_METHOD} --backoff={self.BACK_OFF} trn.t1.cnt t1.lm")
        # os.system("ngramprint --symbols=osyms.t.txt --negativelogs t1.lm t1.probs")
        # make_w2t_mle('t1.probs', out='t1_mle.txt')
        # os.system("fstcompile --isymbols=osyms.t.txt --osymbols=osyms.t.txt --keep_isymbols --keep_osymbols t1_mle.txt t1_mle.bin")
        # os.system("fstinvert t1_mle.bin t1_mle.inv.bin")

    def mle_create_training_data(self):
        os.system("cat isyms.txt osyms.t.txt | cut -f 1 | sort | uniq > msyms.m.lst.txt")
        os.system("ngramsymbols msyms.m.lst.txt msyms.t.txt")
        os.system("cat trn.conll | sed '/^$/d' | awk '{print $2,$1}' > trn.w2t.txt")
        os.system(
            "farcompilestrings --symbols=msyms.t.txt --keep_symbols --unknown_symbol='<UNK>' trn.w2t.txt trn.w2t.far")

    def mle_training(self):
        os.system(f"ngramcount --order={self.NGRAM_ORDER} trn.w2t.far trn.w2t.cnt")
        os.system(f"ngrammake --method={self.SMOOTH_METHOD} --backoff={self.BACK_OFF} trn.w2t.cnt trn.w2t.lm")
        os.system("ngramprint --symbols=msyms.t.txt --negativelogs trn.w2t.lm trn.w2t.probs")

        make_w2t_mle('trn.w2t.probs', out='trn.w2t_mle.txt')
        os.system(
            "fstcompile --isymbols=osyms.t.txt --osymbols=isyms.txt --keep_isymbols --keep_osymbols trn.w2t_mle.txt w2t_mle.bin")
        os.system("fstinvert w2t_mle.bin w2t_mle.inv.bin")

    def mle_evaluate(self):
        os.system("bash ./bin/evaluate_mle.bash")

        refs = read_corpus_conll('tst.conll')
        hyps = read_fst4conll('w2t_t.t1.mle_full.out')

        results = evaluate(refs, hyps)

        pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3)
        pd_tbl.to_csv(
            f"./result/{self.ALG_TYPE}_{self.SMOOTH_METHOD}_{self.BACK_OFF}_{self.NGRAM_ORDER}{self.DATA_TYPE}.csv")

    def run(self):
        print(self.ALG_TYPE, self.DATA_TYPE, self.NGRAM_ORDER, self.SMOOTH_METHOD)
        self.prepare_input()
        self.generate_output()
        self.create_test_set()
        self.lm_create_training_data()
        self.lm_train_unigram()
        self.mle_create_training_data()
        self.mle_training()
        self.mle_evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_type", nargs='?', default='mle')
    parser.add_argument("--data_type", nargs='?', default='')
    parser.add_argument("--ngram_order", nargs='?', type=int, default=1)
    parser.add_argument("--back_off", nargs='?', default='false')
    parser.add_argument("--smooth_method", nargs='?', default='witten_bell')

    args = parser.parse_args()
    model = WfsmMle(args.alg_type, args.data_type, args.ngram_order, args.back_off, args.smooth_method)
    model.run()
