find $1 -name "performc-merge-r.csv" | xargs grep "$2" | sed -e "s/$1\///g" -e "s/_r_0\/performc-merge-r.csv:$2//g"

