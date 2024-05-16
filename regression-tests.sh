#!/bin/sh

echo "\nHello, world! Let's check that our PLAID constraint model continues to behave as expected...\n"
#Reorder by time!
myUnitTests=( 'pl-example01' '5'
	      'pl-example02' '1'
	      'pl-example03' '1'
	      'pl-example04-jonne-doubled' '320'
	      'pl-example05' '1'
	      'pl-example06' '1'
	      'pl-example07' '1'
	      'pl-example08' '3'
	      'pl-example09' '2'
	      'pl-example10' '10'
	      'pl-example11' '385'
	      'pl-example12' '1'
	      'pl-example13' '5'
	      'pl-example14' '1'
	      'pl-example15' '1'
	      'pl-example16' '1' #Currently not supported
	      'pl-example17' '1'
	      'pl-example18' '1'
	      'pl-example19' '5'
	      'pl-example21' '1'
	      'pl-example22' '1'
	      #'pl-example23' '' #Duplicated example
	      'pl-example24' '330'
	      'pl-example25' '10'
	      'pl-example26' '1' #Invalid datafile
	      'pl-example27' '5'
	      'pl-example28' '10'
	      'pl-example29' '5'
	      'pl-example30' '5'
	      'pl-example31' '1'
	      'pl-example35' '60'
	      'pl-example36' '1'
	      'pl-example37' '1'
	      'pl-example38' '1'
	      'pl-example39' '1'
	      'pl-example42' '300'
	      'pl-example43' '1'
	      'pl-example44' '150' #Testing sorted_compounds
	      'pl-example45' '15'
	      'pl-example49' '17' # 1 384-well plates
	      'pl-example53' '1' # 1 384-well plates. Many controls
	      'pl-example56' '75' # 1 384-well plates. Many controls
	      'pl-example46' '330'
	      'pl-example47' '1410' # 3 384-well plates
	      'pl-example48' '1922' # 4 384-well plates
	      '2020-09-30-jonne-slack' '20'
	      '2020-10-08-jonne-slack' '45'
	      '2020-11-13-jonne-slack' '3'
	      'compounds-10-9-3' '20'
	      'dose-response-20-3-1' '30'
	      'dose-response-20-3-2' '30'
	      'dose-response-20-3-3' '30'
	      'screening-8-8-1' '20'
	      'pl-example20' '1000'
	      #'pl-example34' '4685' #MiniZinc ERROR
	      #'pl-example33' '-1'
	      #'pl-example32' '-1'
	    )

len=${#myUnitTests[@]}

for (( i=0; i<$len; i=i+2 ))
do
    echo "Testing file ${myUnitTests[${i}]}.dzn"
    
    read -r line < regression-tests-expected-output/${myUnitTests[${i}]}.txt
    
    echo "Expecting $line"
    
    echo "Usually takes about ${myUnitTests[${i}+1]} sec."

    SECONDS=0

    # Deterministic solution
#    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design.mzn ${myUnitTests[${i}]}.dzn --cmdline-data "testing=true"  &> ${myUnitTests[${i}]}.txt

    # Random and multi-thread
    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design.mzn regression-tests/${myUnitTests[${i}]}.dzn -p 10 -r $RANDOM --cmdline-data "testing=true"  &> regression-tests-results/${myUnitTests[${i}]}.txt

    echo "It took about $SECONDS sec."
    
    read -r result_line < regression-tests-results/${myUnitTests[${i}]}.txt
    
    set -- ${myUnitTests[${i}+1]} 

    
    # Check that we have the expected number of plates or expected error
    
    if [[ $result_line == $line || $1 == "assert" ]]
    then
	echo "We are happy with the number of plates in ${myUnitTests[${i}]}.txt :-)"
    else
	echo "ERROR: The number of plates in ${myUnitTests[${i}]}.dzn changed!!!!!!!!"
	break
    fi   

    # Compare that the solution is the same as before (so hopefully we haven't lost any solutions)
    
    #    if cmp "${myUnitTests[${i}]}.txt" "./output-regression-tests/${myUnitTests[${i}]}.txt";
    #   then
    #	echo "The solution has not changed :-)"
    #   else
    #	echo "WARNING: Check ${myUnitTests[${i}]}.dzn! Something changed!!!!!!!!\n"
    #   fi

    echo "\n"
done

echo "We're done now! Bye!"
