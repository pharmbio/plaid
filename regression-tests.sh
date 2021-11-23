#!/bin/sh
 
echo "Hello, world!"

myUnitTests=( 'pl-example01' '4 plates'
	      'pl-example02' 'MiniZinc: evaluation error:'
	      'pl-example03' 'MiniZinc: evaluation error:'
	      'pl-example04-jonne-doubled' '2 plates'
	      'pl-example05' '2 plates'
	      'pl-example06' '2 plates'
	      'pl-example07-tiny' '4 plates'
	      'pl-example08-small' '2 plates'
	      'pl-example09' '2 plates'
	      'pl-example10' '4 plates'
	      'pl-example11' '2 plates'
	      'pl-example12' '2 plates'
	      'pl-example13' '1 plates'
	      'pl-example14' '1 plates'
	      'pl-example15' '1 plates'
	      'pl-example17' '1 plates'
	      'pl-example18' '1 plates'
	      'pl-example19' '1 plates'
	      'pl-example21' '1 plates'
	      'pl-example22' '4 plates'
	      #'pl-example23' '2 plates' #Duplicated example
	      'pl-example24' '2 plates'
	      'pl-example25' '2 plates'
	      'pl-example27' '3 plates'
	      'pl-example28' '4 plates'
	      'pl-example29' '2 plates'
	      'pl-example30' '1 plates'
	      'pl-example31' 'MiniZinc: evaluation error:'
	      'pl-example35' '1 plates'
	      'pl-example36' 'MiniZinc: evaluation error:'
	      'pl-example37' '1 plates'
	      '2020-10-08-jonne-slack' '1 plates'
	      '2020-11-13-jonne-slack' '1 plates'
	      #'compounds-10-9-3' #'1 plates' #'compounds-10-9-3'
	      'dose-response-20-3-1' '1 plates'
	      'dose-response-20-3-2' '1 plates'
	      'dose-response-20-3-3' '1 plates'
	      'pl-example42' '1 plates'
	      'pl-example20' '1 plates'
	      #'2020-09-30-jonne-slack' #'1 plates'
	    )

len=${#myUnitTests[@]}

for (( i=0; i<$len; i=i+2 ))
do
    echo "Testing file ${myUnitTests[${i}]}.dzn"
    echo "Expecting ${myUnitTests[${i}+1]}"

    SECONDS=0
    
    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design.mzn ${myUnitTests[${i}]}.dzn --cmdline-data "testing=true"  &> ${myUnitTests[${i}]}.txt
    
    echo "It took about $SECONDS sec."
    
    read -r line < ${myUnitTests[${i}]}.txt
    
    set -- ${myUnitTests[${i}+1]} 

    
    # Check that we have the expected number of plates or expected error
    
    if [[ $line == ${myUnitTests[${i}+1]} || $1 == "assert" ]]
    then
	echo "We are happy with the number of plates in ${myUnitTests[${i}]}.txt :-)"
    else
	echo "ERROR: The number of plates in ${myUnitTests[${i}]}.dzn changed!!!!!!!!"
	break
    fi   

    # Compare that the solution is the same as before (so hopefully we haven't lost any solutions)
    
    if cmp "${myUnitTests[${i}]}.txt" "./output-regression-tests/${myUnitTests[${i}]}.txt";
    then
	echo "The solution has not changed :-)\n"
    else
	echo "WARNING: Check ${myUnitTests[${i}]}.dzn! Something changed!!!!!!!!\n"
    fi

done

echo "We're done now! Bye!"
