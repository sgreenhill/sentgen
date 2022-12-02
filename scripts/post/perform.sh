for DIR in $1/*VAE*; do
    python -m Perform $DIR --predictBase cp --output $DIR/perform.pdf --saveCSV $DIR/perform.csv
done
