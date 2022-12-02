#!/bin/bash
gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=$2 -dLastPage=$2 -sOutputFile=$3 $1
