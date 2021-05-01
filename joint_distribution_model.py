import os
from utils import *
from conll import evaluate
import pandas as pd

DATA_TYPE = ""

os.system(f'cp ./dataset/NL2SparQL4NLU.train{DATA_TYPE}.utterances.txt trn.txt')
os.system(f'cp ./dataset/NL2SparQL4NLU.test{DATA_TYPE}.utterances.txt tst.txt')
os.system(f'cp ./dataset/NL2SparQL4NLU.train{DATA_TYPE}.conll.txt trn.conll')
os.system(f'cp ./dataset/NL2SparQL4NLU.test{DATA_TYPE}.conll.txt tst.conll')

trn = read_corpus_conll('trn.conll')
wt_sents = [["+".join(w) for w in s] for s in trn]
wt_osyms = cutoff(wt_sents, tf_min=0)
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

os.system("farcompilestrings --symbols=osyms.wt.txt --keep_symbols --unknown_symbol='<UNK>' trn.wt.txt trn.wt.far")
os.system("ngramcount --order=2 trn.wt.far trn.wt.cnt")
os.system("ngrammake trn.wt.cnt wt2.lm")
os.system("ngramprint --symbols=msyms.t.txt --negativelogs wt2.lm w2t.probs")

make_w2t_wt('osyms.wt.txt', out='w2wt_wt.txt')

os.system("fstcompile --isymbols=isyms.wt.txt --osymbols=osyms.wt.txt --keep_isymbols --keep_osymbols w2wt_wt.txt w2wt_wt.bin")
os.system("farcompilestrings --symbols=isyms.wt.txt --keep_symbols --unknown_symbol='<UNK>' tst.txt tst.wt.far")
os.system("mkdir -p wdir_wt")
os.system('farextract --filename_prefix="wdir_wt/" tst.wt.far')
# os.system("cp wdir_wt/tst.txt-0001 sent.wt.fsa")
# os.system("fstprint sent.wt.fsa")
# os.system("fstcompose sent.wt.fsa w2wt_wt.bin | fstcompose - wt2.lm | fstshortestpath | fstrmepsilon | fsttopsort | fstprint")

os.system('bash ./bin/evaluate_wt.bash')

refs = read_corpus_conll('tst.conll')
hyps = read_fst4conll('w2wt_wt.wt2.out', split=True)

results = evaluate(refs, hyps)

pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3)
pd_tbl.to_csv(f"./result/joint_dist_model.csv")
