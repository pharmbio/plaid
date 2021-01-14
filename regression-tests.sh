#!/bin/sh
 
echo "Hello, world!"

echo "Remember to set testing = true in plate-design.mzn"

myUnitTests=( 'pl-example01' 'pl-example02' 'pl-example03' 'pl-example04-jonne-doubled' 'pl-example05' 'pl-example06' 'pl-example07-tiny' 'pl-example08-small' 'pl-example09' '2020-11-13-jonne-slack' 'pl-example10' 'pl-example11' 'pl-example12' 'pl-example20' '2020-09-30-jonne-slack' '2020-10-08-jonne-slack' )

myUnitTestsResults=( '4 plates' 'assert error E01' 'assert error E01' '2 plates' '2 plates' '2 plates' '4 plates' '2 plates' '2 plates' '1 plates' '4 plates' '2 plates' '2 plates' '1 plates' '1 plates' '1 plates' )

len=${#myUnitTests[@]}

for (( i=0; i<$len; i++ ))
do
    echo "Testing file ${myUnitTests[${i}]}.dzn"
    echo "Expecting ${myUnitTestsResults[${i}]}"
    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design.mzn ${myUnitTests[${i}]}.dzn > ${myUnitTests[${i}]}.txt

    read -r line < ${myUnitTests[${i}]}.txt

    set -- ${myUnitTestsResults[${i}]} 

    if [[ $line == ${myUnitTestsResults[${i}]} || $1 == "assert" ]]
    then
	echo "We are happy with the number of plates in ${myUnitTests[${i}]}.txt"
    else
	echo "The number of plates in ${myUnitTests[${i}]}.dzn changed!!!!!!!!!!"
    fi   

	
    if cmp "${myUnitTests[${i}]}.txt" "./output-regression-tests/${myUnitTests[${i}]}.txt";
    then
	echo ":-)"
    else
	echo "Check ${myUnitTests[${i}]}.dzn! Something changed!!!!!!!!!!"
    fi

    echo ""
    
done
