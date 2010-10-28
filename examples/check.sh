#!/bin/bash

set -e
for d in ripley.libsvm iris.libsvm.h5.csv auto-mpg.uci.h5 friedman-datasets-fri_c2_100_50.arff friedman-datasets-fri_c2_100_50.arff.h5 
do
	echo -n "checking data set $d"
	for type in .csv #.oct .csv .arff #.libsvm .mat
	do
		echo
		echo -n " $type "

		echo -n 1..
		PYTHONPATH=.. python ../scripts/ml2h5conv $d tmp/${d}${type}
		echo -n 2..
		PYTHONPATH=.. python ../scripts/ml2h5conv tmp/${d}${type} tmp/${d}
		echo -n 3..
		PYTHONPATH=.. python ../scripts/ml2h5conv tmp/${d} tmp/${d}2${type}
		if cmp --quiet tmp/${d}${type} tmp/${d}2${type}
		then
			echo -n OK
		else
			echo FAIL
			exit 1
		fi
	done
	echo
done
