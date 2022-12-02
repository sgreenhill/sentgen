find $1 -name "perform-merge-r.csv" | xargs grep "$2" | sed -e "s/$1\///g" -e "s/_r_0\/perform-merge-r.csv:$2//g"

