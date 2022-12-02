#!/bin/bash
OUT=$1
shift
gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=$OUT $*
