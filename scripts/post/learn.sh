SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
for i in $1/*VAE*; do
  echo $i
  (cd $i; gnuplot -c $SCRIPT_DIR/learn.plot `echo $i | sed -e "s/_/-/g"` ; gnuplot -c $SCRIPT_DIR/learnzc.plot `echo $i | sed -e "s/_/-/g"` )
done
pdfmerge.sh $1/learn.pdf `find $1/*VAE* -name learn.pdf`
pdfmerge.sh $1/learnzc.pdf `find $1/*VAE* -name learnzc.pdf`
