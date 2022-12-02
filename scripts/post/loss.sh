python -m mergeloss $1 --csvName "train_{}.csv" > $1/loss-sent.csv
python -m mergeloss $1 --csvName "train_zc_{}.csv" > $1/loss-zc.csv
