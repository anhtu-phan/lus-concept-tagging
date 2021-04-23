from conll import evaluate
import pandas as pd
import os
from utils import *


os.system('bash ./bin/prepare_input_symbol.bash')

trn_data = read_corpus('./trn.txt')
trn_lex = cutoff(trn_data)

with open('isyms.trn.txt', 'w') as f:
    f.write("\n".join(trn_lex) + "\n")

os.system('ngramsymbols isyms.trn.txt isyms.txt')
os.system('bash ./bin/compile_input_into_far.bash')

types = get_chunks('trn.conll')
with open('osyms.u.lst.txt', 'w') as f:
    # let's add 'O'
    f.write("O" + "\n")
    for c in sorted(list(types)):
        # prefix each type with segmentation information
        f.write("B-"+ c + "\n")
        f.write("I-"+ c + "\n")

os.system('ngramsymbols osyms.u.lst.txt osyms.txt')
os.system('bash ./bin/extract_test_into_files.bash')

# make_w2t('isyms.txt', 'osyms.txt', out='w2t_u.txt')

# os.system('bash ./bin/compile_random_model.bash')
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

tlex = cutoff(tags)
with open('osyms.t.lst.txt', 'w') as f:
    f.write("\n".join(tlex) + "\n")

os.system("bash ./bin/train_output_symbol_priori.bash")
make_w2t('isyms.txt', 'osyms.t.txt', out='w2t_t.txt')

os.system("fstcompile --isymbols=isyms.txt --osymbols=osyms.t.txt --keep_isymbols --keep_osymbols w2t_t.txt w2t_t.bin")
os.system("bash ./bin/run_test_full.bash")

refs = read_corpus_conll('tst.conll')
hyps = read_fst4conll('w2t_t.t1.out')

results = evaluate(refs, hyps)

pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3)
pd_tbl.to_csv("./result/no_stop_word_base_line.csv")

os.system("bash ./bin/run_mle.bash")
make_w2t_mle('trn.w2t.probs', out='trn.w2t_mle.txt')
os.system("fstcompile --isymbols=osyms.t.txt --osymbols=isyms.txt --keep_isymbols --keep_osymbols trn.w2t_mle.txt w2t_mle.bin")
os.system("fstinvert w2t_mle.bin w2t_mle.inv.bin")
os.system("bash ./bin/run_test_mle.bash")

refs = read_corpus_conll('tst.conll')
hyps = read_fst4conll('w2t_t.t1.mle_full.out')

results = evaluate(refs, hyps)

pd_tbl = pd.DataFrame().from_dict(results, orient='index').round(decimals=3)
pd_tbl.to_csv("./result/no_stop_word_mle.csv")
