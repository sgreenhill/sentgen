perform-merge-r.sh $1
performc-merge-r.sh $1
gather2.sh $1 20 $1/target-merge.pdf
gather2c.sh $1 20 $1/targetc-merge.pdf
gather2.sh $1 7 $1/rhythm-bar1-merge.pdf
gather2.sh $1 9 $1/rhythm-bar2-merge.pdf
table.sh $1 c:r^2 > $1/table-r2.csv
table.sh $1 valid > $1/table-valid.csv
tablec.sh $1 c:r^2 > $1/tablec-r2.csv
tablec.sh $1 valid > $1/tablec-valid.csv
perform.sh $1
gather.sh $1 20 $1/target.pdf
gather.sh $1 7 $1/rhythm-bar1.pdf
gather.sh $1 9 $1/rhythm-bar2.pdf
dist.sh $1
source.sh $1
space.sh $1
learn.sh $1
dmplot.sh $1
