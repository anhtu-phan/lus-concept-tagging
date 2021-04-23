ngramsymbols osyms.t.lst.txt osyms.t.txt

farcompilestrings \
  --symbols=osyms.t.txt \
  --keep_symbols \
  --unknown_symbol='<UNK>' \
  trn.t.txt trn.t.far

ngramcount --order=1 trn.t.far trn.t1.cnt
ngrammake trn.t1.cnt t1.lm
ngraminfo t1.lm