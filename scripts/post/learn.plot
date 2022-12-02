# set terminal pdf size 40cm, 20.0cm
# set output "plot.pdf"

set terminal pdf size 30cm, 30cm
set output "learn.pdf"
set multiplot layout 3, 3 title ARG1
do for [i=0:8] {
  file = sprintf("train_%d.csv", i)
  set logscale y
  set title sprintf("Attribute %d", i)
  plot file using 1:2 with lines title "Total Loss", file using 1:3 with lines title "Loss R", file using 1:4 with lines title "Loss I", file using 1:5 with lines title "Loss C", file using 1:6 with lines title "Best Loss"
}
