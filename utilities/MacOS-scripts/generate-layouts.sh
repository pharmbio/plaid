#!/bin/sh

echo "\nHello, world! Let's generate some plate layouts...\n"

# List of .dzn files you want to run *without* the extension
myLayoutFiles=( 'pl-example01' 'pl-example05' )

total=4

len=${#myLayoutFiles[@]}

for (( j=0; j<$total; j=j+1 ))
do
    for (( i=0; i<$len; i=i+1 ))
    do
	echo "Generating a layout for file ${myLayoutFiles[${i}]}.dzn"
	
	SECONDS=0

	# Random and multi-thread
	/Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode ../../plate-design.mzn ${myLayoutFiles[${i}]}.dzn -p 10 -r $RANDOM  &> ${myLayoutFiles[${i}]}-${j}.csv

	echo "It took about $SECONDS sec."	

	echo "\n"
    done
done

echo "We're done now! Bye!"
