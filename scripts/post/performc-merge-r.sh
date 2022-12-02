for DIR in $1/*VAE*_r_0 ; do
	echo $DIR
    python -m Perform ${DIR//r_0/r_*} --predictBase cp --output $DIR/performc-merge-r.pdf --saveCSV $DIR/performc-merge-r.csv --categoricalMetric True
done
