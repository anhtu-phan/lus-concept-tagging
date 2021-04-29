from conll import evaluate
import pandas as pd
import os
from utils import *


DATA_TYPE = "_no_stop_word"
NGRAM_ORDER = 6
BACK_OFF = 'true'
SMOOTH_METHOD = "witten_bell"

os.system(f'cp ./dataset/NL2SparQL4NLU.train{DATA_TYPE}.utterances.txt trn.txt')
os.system(f'cp ./dataset/NL2SparQL4NLU.test{DATA_TYPE}.utterances.txt tst.txt')
os.system(f'cp ./dataset/NL2SparQL4NLU.train{DATA_TYPE}.conll.txt trn.conll')
os.system(f'cp ./dataset/NL2SparQL4NLU.test{DATA_TYPE}.conll.txt tst.conll')

trn_data = read_corpus('./trn.txt')
trn_lex = cutoff(trn_data)

with open('isyms.trn.txt', 'w') as f:
    f.write("\n".join(trn_lex) + "\n")

os.system('ngramsymbols isyms.trn.txt isyms.txt')
os.system("farcompilestrings --symbols=isyms.txt --keep_symbols --unknown_symbol='<UNK>' trn.txt trn.far")
os.system("farcompilestrings --symbols=isyms.txt --keep_symbols --unknown_symbol='<UNK>' tst.txt tst.far")

types = get_chunks('trn.conll')
with open('osyms.u.lst.txt', 'w') as f:
    # let's add 'O'
    f.write("O" + "\n")
    for c in sorted(list(types)):
        # prefix each type with segmentation information
        f.write("B-"+ c + "\n")
        f.write("I-"+ c + "\n")

os.system('ngramsymbols osyms.u.lst.txt osyms.txt')
os.system('mkdir -p wdir')
os.system('farextract --filename_prefix="wdir/" tst.far')

# make_w2t('isyms.txt', 'osyms.txt', out='w2t_u.txt')

# os.system('fstcompile --isymbols=isyms.txt --osymbols=osyms.txt --keep_isymbols --keep_osymbols w2t_u.txt w2t_u.bin')
# os.system('bash ./bin/run_test.bash')


# refs = read_corpus_conll('tst.conll')
# hyps = read_fst4conll('w2t_u.out')
# results = evaluate(refs, hyps)
#
# pd_tbl = pd.DataFrame().from_dict(results, orient='index')
# pd_tbl.round(decimals=3)
# print(pd_tbl)
#
trn = read_corpus_conll('trn.conll')
tags = get_column(trn, column=-1)
# write data
with open('trn.t.txt', 'w') as f:
    for s in tags:
        f.write(" ".join(s) + "\n")

# tlex = cutoff(tags)
# with open('osyms.t.lst.txt', 'w') as f:
#     f.write("\n".join(tlex) + "\n")

os.system("ngramsymbols osyms.u.lst.txt osyms.t.txt")
os.system("farcompilestrings --symbols=osyms.t.txt --keep_symbols --unknown_symbol='<UNK>' trn.t.txt trn.t.far")
os.system(f"ngramcount --order={NGRAM_ORDER} trn.t.far trn.t1.cnt")
os.system(f"ngrammake --method={SMOOTH_METHOD} --backoff={BACK_OFF} trn.t1.cnt t1.lm")

# make_w2t('isyms.txt', 'osyms.t.txt', out='w2t_t.txt')
#
# os.system("fstcompile --isymbols=isyms.txt --osymbols=osyms.t.txt --keep_isymbols --keep_osymbols w2t_t.txt w2t_t.bin")
# os.system("bash ./bin/run_test_full.bash")
#
# refs = read_corpus_conll('tst.conll')
# hyps = read_fst4conll('w2t_t.t1.out')
#
# results = evaluate(refs, hyps)
# pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3)
# pd_tbl.to_csv("./result/"+RUN_TYPE+"_base_line.csv")

os.system("cat isyms.txt osyms.t.txt | cut -f 1 | sort | uniq > msyms.m.lst.txt")
os.system("ngramsymbols msyms.m.lst.txt msyms.t.txt")
os.system("cat trn.conll | sed '/^$/d' | awk '{print $2,$1}' > trn.w2t.txt")
os.system("farcompilestrings --symbols=msyms.t.txt --keep_symbols --unknown_symbol='<UNK>' trn.w2t.txt trn.w2t.far")
os.system(f"ngramcount --order={NGRAM_ORDER} trn.w2t.far trn.w2t.cnt")
os.system(f"ngrammake --method={SMOOTH_METHOD} --backoff={BACK_OFF} trn.w2t.cnt trn.w2t.lm")

make_w2t_mle('trn.w2t.probs', out='trn.w2t_mle.txt')
os.system("fstcompile --isymbols=osyms.t.txt --osymbols=isyms.txt --keep_isymbols --keep_osymbols trn.w2t_mle.txt w2t_mle.bin")
os.system("fstinvert w2t_mle.bin w2t_mle.inv.bin")
os.system("bash ./bin/evaluatate.bash")

refs = read_corpus_conll('tst.conll')
hyps = read_fst4conll('w2t_t.t1.mle_full.out')

results = evaluate(refs, hyps)

pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3)
pd_tbl.to_csv(f"./result/{SMOOTH_METHOD}_{BACK_OFF}_{NGRAM_ORDER}{DATA_TYPE}.csv")
