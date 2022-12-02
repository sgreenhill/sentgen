SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
for i in $1/*VAE*; do
  ( cd $i; gnuplot $SCRIPT_DIR/dmelodies.plot )
done
for i in 1 2 3 4 5 6 7 8 9 ; do
  echo $i
  convert `find $1/*VAE* -name dmplot-$i.png` $1/dmplot-$i.pdf
done
