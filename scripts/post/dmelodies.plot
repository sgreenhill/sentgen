# set terminal pdf size 40cm, 20.0cm
# set output "plot.pdf"

set terminal pngcairo size 2000, 1000
set datafile separator ","
do for [i=1:9] {
  set output sprintf("dmplot-%d.png", i)
  print(i)
  attrFile = sprintf('sp_%d.csv', i-1)
  zFile = sprintf('z2_%d.csv', i-1)
  set multiplot layout 2, 4
  do for [j=1:9] {
    if (i != j) {
      plot sprintf("< (paste %s %s | sed -e 's/\t/,/g')", attrFile, zFile) using i:j:(column(i+32)-column(i)):(column(j+32)-column(j)) with vectors nohead notitle, zFile using i : j lt 7 lc rgb '#ff0000' title sprintf("Target %d - %d", i, j), attrFile using i : j lc rgb '#00ff00' title sprintf("Prediction %d - %d", i, j)
    }
  }
  unset multiplot
}
