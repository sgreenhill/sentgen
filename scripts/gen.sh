#!/bin/bash

echo Args: $*

SPLIT=0.7,0.05,0.25
THREADS=1
JOBS=4
PAIRS=1000
GEN=data/gen
THRESH=0.00
PREPDIR=identity

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --split)
      SPLIT="$2"
      shift # past argument
      shift # past value
      ;;
    --threads)
      THREADS="$2"
      shift # past argument
      shift # past value
      ;;
    --jobs)
      JOBS="$2"
      shift # past argument
      shift # past value
      ;;
    --pairs)
      PAIRS="$2"
      shift # past argument
      shift # past value
      ;;
    --gen)
      GEN="$2"
      shift # past argument
      shift # past value
      ;;
    --prepdir)
      PREPDIR="$2"
      shift # past argument
      shift # past value
      ;;
    --thresh)
      THRESH="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ ! -d $GEN ]; then
	mkdir -p $GEN
    SRC=`python -m splitname $SPLIT`
	ls dmelodies_controllability/src/saved_models-$SRC | parallel --ungroup --jobs $JOBS python -m dmgenerate --generate $GEN --split $SPLIT --threads $THREADS
fi

if [ ! -f findpairs ]; then
	gcc -O3 src/findpairs.c -lm -o findpairs
fi

if [ ! -d $GEN/$PREPDIR ]; then
	echo Preparing data
	for i in 0 1 2; do
	    FILE=$GEN/DMelodiesVAE_RNN_ar-VAE_b_0.2_c_50.0_g_0.1_d_10.0_r_${i}_.npz
		PREP=$GEN/$PREPDIR/$i/work
		mkdir -p $PREP
		# build a temporary repository, and extract the normalised features to a text file
		python -m build init2 $PREP $FILE skip --trainTestPairs 0,0
		python -m build getf $PREP $PREP/f-$i.txt

		# find training pairs using defined threshold
		parallel --line-buffer --jobs $JOBS ./findpairs $PREP/f-$i.txt $THRESH $PAIRS --prefix $PREP --attribute {} $* ::: 0 1 2 3 4 5 6 7 8

		# import pairs files batch into NPZ container
		python -m build importf $PREP $GEN/$PREPDIR/$i/pairs.npz
#		rm -rf $PREP
	done
	echo Done
fi
