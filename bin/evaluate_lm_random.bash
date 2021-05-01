wdir='wdir'
farr=($(ls $wdir))

for f in ${farr[@]}
do
    fstcompose $wdir/$f w2t_u.bin | fstcompose - t1.lm | fstrandgen | fstrmepsilon | fsttopsort | fstprint --isymbols=isyms.txt
done > w2t_u_lm.out