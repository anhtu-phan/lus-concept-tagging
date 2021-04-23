wdir='wdir'
farr=($(ls $wdir))

for f in ${farr[@]}
do
  fstcompose $wdir/$f w2t_u.bin | fstrandgen | fstrmepsilon | fsttopsort | fstprint --isymbols=isyms.txt
done > w2t_u.out