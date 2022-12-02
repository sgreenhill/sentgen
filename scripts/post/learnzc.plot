# set terminal pdf size 40cm, 20.0cm
# set output "plot.pdf"

set terminal pdf size 30cm, 30cm
set output "learnzc.pdf"
set multiplot layout 3, 3 title ARG1
do for [i=0:8] {
  file = sprintf("train_zc_%d.csv", i)
  set logscale y
  set title sprintf("Attribute %d", i)
  plot file using 1:3 with lines title "Total Loss", file using 1:4 with lines title "Best Loss"
}
