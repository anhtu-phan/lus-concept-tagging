wdir='wdir'
farr=($(ls $wdir))

for f in ${farr[@]}
do
  fstcompose $wdir/$f w2t_t.bin | fstcompose - t1.lm |\
  fstshortestpath | fstrmepsilon | fsttopsort | fstprint --isymbols=isyms.txt
done > w2t_t.t1.out