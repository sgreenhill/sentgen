# set terminal pdf size 40cm, 20.0cm
# set output "plot.pdf"

set terminal pngcairo size 2000, 1000
do for [i=1:9] {
  set output sprintf("spaceplot-%d.png", i)
  print(i)
  set multiplot layout 2, 4 title sprintf("Z plot for dimension %d of %s", i, ARG1)
  do for [j=1:9] {
    if (i != j) {

      set xlabel sprintf("Dimension %d", i)
      set ylabel sprintf("Dimension %d", j)
      plot "< paste ztmp.csv ftmp.csv" using i:j:(column(32+i)) lt 7 lc variable notitle
    }
  }
  unset multiplot
}
