#!/bin/bash

set -e
for d in friedman-datasets-fri_c2_100_50.arff iris.libsvm.h5.csv friedman-datasets-fri_c2_100_50.arff.h5 auto-mpg.uci.h5 
do
	echo -n "checking data set $d"
	for type in .oct .csv .mat #.arff 
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
			if test "$type" = '.mat'
			then
				echo "load tmp/${d}${type}; save tmp/${d}${type}.oct" | octave -q
				echo "load tmp/${d}2${type}; save tmp/${d}2${type}.oct" | octave -q
				grep -v '# Created by Octave' tmp/${d}${type}.oct >tmp/${d}${type}.oct2
				grep -v '# Created by Octave' tmp/${d}2${type}.oct >tmp/${d}2${type}.oct2

				if cmp --quiet tmp/${d}${type}.oct2 tmp/${d}2${type}.oct2
				then
					echo -n OK
				else
					echo FAIL
					exit 1
				fi
			else
				echo FAIL
				exit 1
			fi
		fi
	done
	echo
done

for d in ripley.libsvm
do
	echo -n "checking data set $d"
	for type in .libsvm .oct 
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
