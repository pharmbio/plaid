#!/bin/sh

### Write here the name of the file with your multiplate experiment *without* the extension ###
myBigExperimentFile='pl-example01'

echo "\nHello, world! Let's generate some plate layouts...\n"

echo "\nFirst, we are going to split the experiment into individual plates...\n"

/Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode ../../plate-split/plate-split.mzn $myBigExperimentFile.dzn -p 10 -r $RANDOM --soln-sep '' &> $myBigExperimentFile.txt

split -d -p '^%%% Experiment.*' $myBigExperimentFile.txt $myBigExperimentFile-plate-

myLayoutFiles=( $myBigExperimentFile-plate-[0-9][0-9]* )

total=4

len=${#myLayoutFiles[@]}

echo "\nNow we are going to generate $total layouts for each of the $len individual plates...\n"


for file in $myBigExperimentFile-plate-[0-9][0-9]*
do
    mv "$file" "$file.dzn"
done


for (( j=0; j<$total; j=j+1 ))
do
    for (( i=0; i<$len; i=i+1 ))
    do
	echo "Generating a layout for file ${myLayoutFiles[${i}]}"
	
	SECONDS=0

	# Random and multi-thread
	/Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode ../../plate-design.mzn --data ${myLayoutFiles[${i}]}.dzn -p 10 -r $RANDOM  &> ${myLayoutFiles[${i}]}-${j}.csv

	echo "It took about $SECONDS sec."	

	echo "\n"
    done
done

echo "We're done now! Bye!"
