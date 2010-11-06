#!/bin/bash

set -e
for d in uci-20070111-zoo.h5 \
	uci-20070111-zoo.arff \
	bodyfat.h5 \
	friedman-datasets-fri_c2_100_50.arff \
	C_elegans_acc_seq.arff \
	iris.libsvm.h5.csv \
	friedman-datasets-fri_c2_100_50.arff.h5 \
	auto-mpg.uci.h5
do
	echo -n "checking data set $d"
	for type in .oct .csv .arff .mat 
	do
		echo
		echo -n "$type "

		echo -n 1.. 
		PYTHONPATH=.. python ../scripts/ml2h5 $d tmp/${d}${type}
		echo -n 2..
		PYTHONPATH=.. python ../scripts/ml2h5 tmp/${d}${type} tmp/${d}
		echo -n 3..
		PYTHONPATH=.. python ../scripts/ml2h5 tmp/${d} tmp/${d}2${type}
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

for d in ripley.libsvm rcv1subset_topics_train_1.svm
do
	echo -n "checking data set $d"
	for type in .oct .mat
	do
		echo
		echo -n " $type "

		echo -n 1..
		PYTHONPATH=.. python ../scripts/ml2h5 $d tmp/${d}${type}
		echo -n 2..
		PYTHONPATH=.. python ../scripts/ml2h5 tmp/${d}${type} tmp/${d}
		echo -n 3..
		PYTHONPATH=.. python ../scripts/ml2h5 tmp/${d} tmp/${d}2${type}
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
