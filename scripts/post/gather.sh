# p1 - base directory
# p2 - page number
# p3 - output PDF file

files=(`find $1/*VAE* -name perform.pdf`)
for i in ${!files[@]}; do
  fromFile=${files[i]}
  toFile=tmp-gather-`printf "%04d" $i`.pdf
  pdfextract.sh $fromFile $2 $toFile
done
pdfmerge.sh $3 tmp-gather-*.pdf
rm tmp-gather-*.pdf
