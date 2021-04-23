farcompilestrings \
    --symbols=isyms.txt \
    --keep_symbols \
    --unknown_symbol='<UNK>' \
    trn.txt trn.far

farcompilestrings \
    --symbols=isyms.txt \
    --keep_symbols \
    --unknown_symbol='<UNK>' \
    tst.txt tst.far