cat isyms.txt osyms.t.txt | cut -f 1 | sort | uniq > msyms.m.lst.txt
ngramsymbols msyms.m.lst.txt msyms.t.txt

# let's convert data to ngrams
cat trn.conll | sed '/^$/d' | awk '{print $2,$1}' > trn.w2t.txt

# compile to far
farcompilestrings \
  --symbols=msyms.t.txt \
  --keep_symbols \
  --unknown_symbol='<UNK>' \
  trn.w2t.txt trn.w2t.far

# count bigrams
ngramcount --order=2 trn.w2t.far trn.w2t.cnt
# make a model
ngrammake trn.w2t.cnt trn.w2t.lm

# print ngram probabilities as negative logs
ngramprint \
  --symbols=msyms.t.txt\
  --negativelogs \
  trn.w2t.lm trn.w2t.probs