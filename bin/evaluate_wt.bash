wdir='wdir_wt'
farr=($(ls $wdir))

for f in ${farr[@]}
do
  fstcompose $wdir/$f w2wt_wt.bin | fstcompose - wt2.lm | fstshortestpath | fstrmepsilon | fsttopsort | fstprint --isymbols=isyms.wt.txt
done > w2wt_wt.wt2.out