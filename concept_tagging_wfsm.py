import argparse
import os

import pandas as pd

from conll import evaluate
from pre_process_data import *
from utils import *


class WfsmMle:
    def __init__(self, alg_type="mle", data_type="", ngram_order_lm=1, ngram_order_wt=2, ngram_order_joint=2,
                 smooth_method_lm='witten_bell', smooth_method_wt='witten_bell', smooth_method_joint='witten_bell'):
        self.ALG_TYPE = alg_type
        self.DATA_TYPE = data_type
        self.NGRAM_ORDER_LM = ngram_order_lm
        self.NGRAM_ORDER_WT = ngram_order_wt
        self.NGRAM_ORDER_JOINT = ngram_order_joint
        self.SMOOTH_METHOD_LM = smooth_method_lm
        self.SMOOTH_METHOD_WT = smooth_method_wt
        self.SMOOTH_METHOD_JOINT = smooth_method_joint

    def prepare_input(self, data_type):
        os.system(f'cp ./dataset/NL2SparQL4NLU.train{data_type}.utterances.txt trn.txt')
        os.system(f'cp ./dataset/NL2SparQL4NLU.test{data_type}.utterances.txt tst.txt')
        os.system(f'cp ./dataset/NL2SparQL4NLU.train{data_type}.conll.txt trn.conll')
        os.system(f'cp ./dataset/NL2SparQL4NLU.test{data_type}.conll.txt tst.conll')

    def mle_prepare_data(self):
        trn_data = read_corpus('trn.txt')
        # if data_type == '':
        trn_lex = cutoff(trn_data)
        # else:
        #     trn_lex = get_lexicon(trn_data)
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

    def create_test_set(self, folder_name):
        os.system(f'mkdir -p {folder_name}')
        os.system(f'farextract --filename_prefix="{folder_name}/" tst.far')

    def lm_create_training_data(self, data_type=''):
        trn = read_corpus_conll('trn.conll')
        tags = get_column(trn, column=-1)
        # write data
        with open('trn.t.txt', 'w') as f:
            for s in tags:
                f.write(" ".join(s) + "\n")
        if data_type == '':
            tlex = cutoff(tags)
            with open('osyms.t.lst.txt', 'w') as f:
                f.write("\n".join(tlex) + "\n")
            os.system("ngramsymbols osyms.t.lst.txt osyms.t.txt")
        else:
            os.system("ngramsymbols osyms.u.lst.txt osyms.t.txt")
        os.system("farcompilestrings --symbols=osyms.t.txt --keep_symbols --unknown_symbol='<UNK>' trn.t.txt trn.t.far")

    def lm_train_unigram(self, ngram_order=1, smooth_method='witten_bell'):
        os.system(f"ngramcount --order={ngram_order} trn.t.far trn.t1.cnt")
        os.system(f"ngrammake --method={smooth_method} trn.t1.cnt t1.lm")
        os.system("ngramprint --symbols=osyms.t.txt --negativelogs t1.lm t1.probs")
        # make_w2t_mle('t1.probs', out='t1_mle.txt')
        # os.system("fstcompile --isymbols=osyms.t.txt --osymbols=osyms.t.txt --keep_isymbols --keep_osymbols t1_mle.txt t1_mle.bin")
        # os.system("fstinvert t1_mle.bin t1_mle.inv.bin")

    def mle_create_training_data(self):
        os.system("cat isyms.txt osyms.t.txt | cut -f 1 | sort | uniq > msyms.m.lst.txt")
        os.system("ngramsymbols msyms.m.lst.txt msyms.t.txt")
        os.system("cat trn.conll | sed '/^$/d' | awk '{print $2,$1}' > trn.w2t.txt")
        os.system(
            "farcompilestrings --symbols=msyms.t.txt --keep_symbols --unknown_symbol='<UNK>' trn.w2t.txt trn.w2t.far")

    def mle_training(self, ngram_order=2, smooth_method='witten_bell'):
        os.system(f"ngramcount --order={ngram_order} trn.w2t.far trn.w2t.cnt")
        os.system(f"ngrammake --method={smooth_method} trn.w2t.cnt trn.w2t.lm")
        os.system("ngramprint --symbols=msyms.t.txt --negativelogs trn.w2t.lm trn.w2t.probs")

        make_w2t_mle('trn.w2t.probs', out='trn.w2t_mle.txt')
        os.system(
            "fstcompile --isymbols=osyms.t.txt --osymbols=isyms.txt --keep_isymbols --keep_osymbols trn.w2t_mle.txt w2t_mle.bin")
        os.system("fstinvert w2t_mle.bin w2t_mle.inv.bin")

    def mle_evaluate(self, file_name):
        os.system("bash ./bin/evaluate_mle.bash")

        refs = read_corpus_conll('tst.conll')
        hyps = read_fst4conll('w2t_t.t1.mle_full.out')

        results = evaluate(refs, hyps)

        pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3).sort_values("s", ascending=False)
        pd_tbl.to_csv(
            f"./result/{file_name}.csv", sep='&')

    def joint_model_prepare_data(self):
        trn = read_corpus_conll('trn.conll')
        wt_sents = [["+".join(w) for w in s] for s in trn]
        wt_osyms = cutoff(wt_sents)
        wt_isyms = [w.split('+')[0] for w in wt_osyms]

        with open('trn.wt.txt', 'w') as f:
            for s in wt_sents:
                f.write(" ".join(s) + "\n")

        with open('osyms.wt.lst.txt', 'w') as f:
            f.write("\n".join(wt_osyms) + "\n")

        with open('isyms.wt.lst.txt', 'w') as f:
            f.write("\n".join(wt_isyms) + "\n")

        os.system("ngramsymbols osyms.wt.lst.txt osyms.wt.txt")
        os.system("ngramsymbols isyms.wt.lst.txt isyms.wt.txt")

        os.system(
            "farcompilestrings --symbols=osyms.wt.txt --keep_symbols --unknown_symbol='<UNK>' trn.wt.txt trn.wt.far")
        os.system("farcompilestrings --symbols=isyms.wt.txt --keep_symbols --unknown_symbol='<UNK>' tst.txt tst.far")

    def train_conceptual_lm(self, ngram_order=2, smooth_method='witten_bell'):
        os.system(f"ngramcount --order={ngram_order} trn.wt.far trn.wt.cnt")
        os.system(f"ngrammake --method={smooth_method} trn.wt.cnt wt2.lm")
        # os.system("ngramprint --symbols=osyms.wt.txt --negativelogs wt2.lm w2t.probs")
        make_w2t_wt('osyms.wt.txt', out='w2wt_wt.txt')
        os.system(
            "fstcompile --isymbols=isyms.wt.txt --osymbols=osyms.wt.txt --keep_isymbols --keep_osymbols w2wt_wt.txt w2wt_wt.bin")

    def evaluate_joint_model(self, file_name):
        os.system('bash ./bin/evaluate_wt.bash')

        refs = read_corpus_conll('tst.conll')
        hyps = read_fst4conll('w2wt_wt.wt2.out', split=True)

        results = evaluate(refs, hyps)

        pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3).sort_values("s", ascending=False)
        pd_tbl.to_csv(f"./result/{file_name}.csv", sep='&')

    def pre_process_data(self):
        if self.DATA_TYPE != '':
            remove_stop_words("dataset/NL2SparQL4NLU.train.utterances.txt",
                              "dataset/NL2SparQL4NLU.train.conll.txt",
                              "dataset/NL2SparQL4NLU.train_no_stop_word.utterances.txt",
                              "dataset/NL2SparQL4NLU.train_no_stop_word.conll.txt")
            remove_stop_words("dataset/NL2SparQL4NLU.test.utterances.txt",
                              "dataset/NL2SparQL4NLU.test.conll.txt",
                              "dataset/NL2SparQL4NLU.test_no_stop_word.utterances.txt",
                              "dataset/NL2SparQL4NLU.test_no_stop_word.conll.txt", training=False)
            if self.DATA_TYPE == '_norm_no_stop_word':
                norm_data_input("dataset/NL2SparQL4NLU.train_no_stop_word.utterances.txt",
                                "dataset/NL2SparQL4NLU.train_norm_no_stop_word.utterances.txt")
                norm_data_input("dataset/NL2SparQL4NLU.test_no_stop_word.utterances.txt",
                                "dataset/NL2SparQL4NLU.test_norm_no_stop_word.utterances.txt")
                norm_data_input("dataset/NL2SparQL4NLU.train_no_stop_word.conll.txt",
                                "dataset/NL2SparQL4NLU.train_norm_no_stop_word.conll.txt",
                                file_type='conll')
                norm_data_input("dataset/NL2SparQL4NLU.test_no_stop_word.conll.txt",
                                "dataset/NL2SparQL4NLU.test_norm_no_stop_word.conll.txt",
                                file_type='conll')

    def run_mle(self):
        self.pre_process_data()
        self.prepare_input(self.DATA_TYPE)
        self.mle_prepare_data()
        self.generate_output()
        self.create_test_set("wdir")
        self.lm_create_training_data(self.DATA_TYPE)
        self.lm_train_unigram(self.NGRAM_ORDER_LM, self.SMOOTH_METHOD_LM)
        self.mle_create_training_data()
        self.mle_training(self.NGRAM_ORDER_WT, self.SMOOTH_METHOD_WT)
        self.mle_evaluate(
            f'{self.ALG_TYPE}{self.DATA_TYPE}_{self.SMOOTH_METHOD_LM}_{self.SMOOTH_METHOD_WT}_{self.NGRAM_ORDER_LM}_{self.NGRAM_ORDER_WT}')

    def run_joint_model(self):
        self.pre_process_data()
        self.prepare_input(self.DATA_TYPE)
        self.joint_model_prepare_data()
        self.create_test_set("wdir_wt")
        self.train_conceptual_lm(self.NGRAM_ORDER_JOINT, self.SMOOTH_METHOD_JOINT)
        self.evaluate_joint_model(
            f'{self.ALG_TYPE}{self.DATA_TYPE}_{self.SMOOTH_METHOD_JOINT}_{self.NGRAM_ORDER_JOINT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_type", nargs='?', default='mle')
    parser.add_argument("--data_type", nargs='?', default='')
    parser.add_argument("--ngram_order_lm", nargs='?', type=int, default=1)
    parser.add_argument("--ngram_order_wt", nargs='?', type=int, default=2)
    parser.add_argument("--ngram_order_joint", nargs='?', type=int, default=2)
    parser.add_argument("--smooth_method_lm", nargs='?', default='witten_bell')
    parser.add_argument("--smooth_method_wt", nargs='?', default='witten_bell')
    parser.add_argument("--smooth_method_joint", nargs='?', default='witten_bell')

    if not os.path.exists("./result"):
        os.system("mkdir result")

    args = parser.parse_args()
    model = WfsmMle(args.alg_type, args.data_type, args.ngram_order_lm, args.ngram_order_wt, args.ngram_order_joint,
                    args.smooth_method_lm, args.smooth_method_wt, args.smooth_method_joint)
    if args.alg_type == 'mle':
        model.run_mle()
    elif args.alg_type == 'joint':
        model.run_joint_model()
    else:
        raise ValueError(f"DOES NOT SUPPORT {args.alg_type} ALGORITHM")
