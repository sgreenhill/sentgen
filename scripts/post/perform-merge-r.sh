for DIR in $1/*VAE*_r_0 ; do
	echo $DIR
    python -m Perform ${DIR//r_0/r_*} --predictBase cp --output $DIR/perform-merge-r.pdf --saveCSV $DIR/perform-merge-r.csv
done
