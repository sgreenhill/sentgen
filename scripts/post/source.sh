for i in 0 1 2; do
	python -m Aux plotsource $1/ar-VAE_b_0.2_c_50.0_g_0.1_d_10.0_r_$i $1/source-train-$i.pdf --train True --title "$1 Train seed=$i"
	python -m Aux plotsource $1/ar-VAE_b_0.2_c_50.0_g_0.1_d_10.0_r_$i $1/source-test-$i.pdf --train False --title "$1 Test seed=$i"
done
