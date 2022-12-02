files=(`find $1/*VAE* -name perform-merge-r.pdf`)
for i in ${!files[@]}; do
  fromFile=${files[i]}
  toFile=tmp-gather-`printf "%04d" $i`.pdf
  pdfextract.sh $fromFile $2 $toFile
done
pdfmerge.sh $3 tmp-gather-*.pdf
rm tmp-gather-*.pdf
