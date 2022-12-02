SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
for i in $1/*VAE*; do
  echo $i
  python -m build printz $i --limit 5000 | sed -e "s/,/ /g" > $i/ztmp.csv
  python -m build printf $i --limit 5000 --denormalise True | sed -e "s/,/ /g" > $i/ftmp.csv
  (cd $i; gnuplot -c $SCRIPT_DIR/space.plot `echo $i | sed -e "s/_/-/g"` )
  rm $i/ftmp.csv
  rm $i/ztmp.csv
done
for i in 1 2 3 4 5 6 7 8 9; do
  convert `find $1/*VAE* -name spaceplot-$i.png` $1/spaceplot-$i.pdf
done
