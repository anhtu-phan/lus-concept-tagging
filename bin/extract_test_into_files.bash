wdir='wdir'
mkdir -p $wdir

farextract --filename_prefix="$wdir/" tst.far
cp $wdir/tst.txt-0001 sent.fsa

#fstprint sent.fsa