wdir='wdir'
farr=($(ls $wdir))

for f in ${farr[@]}
do
#  fstcompose $wdir/$f w2t_mle.inv.bin | fstshortestpath | fstrmepsilon | fsttopsort | fstprint
  fstcompose $wdir/$f w2t_mle.inv.bin | fstcompose - t1.lm |  fstshortestpath | fstrmepsilon | fsttopsort | fstprint
#  fstcompose $wdir/$f w2t_mle.inv.bin | fstcompose - t1_mle.inv.bin | fstshortestpath | fstrmepsilon | fsttopsort | fstprint
done > w2t_t.t1.mle_full.out