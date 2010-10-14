#!/bin/bash

set -e
for d in diabetes_scale.libsvm.h5 friedman-datasets-fri_c2_100_50.arff.h5 iris.libsvm.h5.csv #auto-mpg.uci.h5 
do
	echo -n "checking data set $d"
	for type in .csv #.arff .libsvm .arff .oct .mat
	do
		echo -n " $type "

		PYTHONPATH=.. python ../scripts/ml2h5conv $d tmp/${d}${type}
		PYTHONPATH=.. python ../scripts/ml2h5conv tmp/${d}${type} tmp/${d}
		PYTHONPATH=.. python ../scripts/ml2h5conv tmp/${d} tmp/${d}2${type}
		if cmp --quiet tmp/${d}${type} tmp/${d}2${type}
		then
			echo -n OK
		else
			echo error
			exit 1
		fi
	done
	echo
done
