for k in 0 1 2 3 4 5 6 7 8; do
for i in $1/*VAE*; do
  echo $i
  python -m Aux plotspace $i/fit_$k.csv $i/dist_$k.pdf --title `echo $i | sed -e "s/_/-/g"`
done
pdfmerge.sh $1/dist_$k.pdf `find $1/*VAE* -name "dist_$k.pdf"`
done
