dpath='./dataset/NL2SparQL4NLU'

#cp $dpath.train.utterances.txt trn.txt
#cp $dpath.test.utterances.txt tst.txt
cp $dpath.train_no_stop_word.utterances.txt trn.txt
cp $dpath.test_no_stop_word.utterances.txt tst.txt

cp $dpath.train_no_stop_word.conll.txt trn.conll
cp $dpath.test_no_stop_word.conll.txt tst.conll
#cp $dpath.train.conll.txt trn.conll
#cp $dpath.test.conll.txt tst.conll