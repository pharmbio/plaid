#!/bin/sh

echo "\nHello, world! Let's check if our PLAID constraint model continues to behave as expected...\n"

myUnitTests=( 'pl-example01' '4 plates' '3'
	      'pl-example02' 'Error: assertion failed: Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E01)' '1'
	      'pl-example03' 'Error: assertion failed: Invalid data: the design is unsatisfiable. It is not possible to divide the compounds and controls evenly across the plates. (E01)' '1'
	      'pl-example05' '2 plates' '1'
	      'pl-example06' '2 plates' '1'
	      'pl-example07-tiny' '4 plates' '1'
	      'pl-example08-small' '2 plates' '3'
	      'pl-example09' '2 plates' '2'
	      'pl-example10' '4 plates' '14'
	      'pl-example11' '2 plates' '384'
	      'pl-example12' '2 plates' '1'
	      'pl-example13' '1 plates' '5'
	      'pl-example14' '1 plates' '1'
	      'pl-example15' '1 plates' '1'
	      'pl-example17' '1 plates' '1'
	      'pl-example18' '1 plates' '1'
	      'pl-example19' '1 plates' '3'
	      'pl-example21' '1 plates' '1'
	      'pl-example22' '4 plates' '1'
	      #'pl-example23' '2 plates' '' #Duplicated example
	      'pl-example24' '2 plates' '332'
	      'pl-example25' '2 plates' '12'
	      'pl-example27' '3 plates' '2'
	      'pl-example28' '4 plates' '9'
	      'pl-example29' '2 plates' '4'
	      'pl-example30' '1 plates' '5'
	      'pl-example31' 'Error: assertion failed: Invalid datafile: There are too many controls of only one kind. This is not allowed at the moment. If you believe this is a mistake, please contact the developers.' '1'
	      'pl-example35' '1 plates' '66'
	      'pl-example36' 'Error: assertion failed: Invalid datafile: There are too many controls of only one kind. This is not allowed at the moment. If you believe this is a mistake, please contact the developers.' '1'
	      'pl-example37' '1 plates' '1'
	      'pl-example44' '2 plates' '150' #Testing sorted_compounds
	      '2020-10-08-jonne-slack' '1 plates' '21'
	      '2020-11-13-jonne-slack' '1 plates' '3'
	      #'compounds-10-9-3' #'1 plates' '' #'compounds-10-9-3'
	      'dose-response-20-3-1' '1 plates' '115'
	      'dose-response-20-3-2' '1 plates' '121'
	      'dose-response-20-3-3' '1 plates' '118'
	      'screening-8-8-1' '1 plates' '20'
	      'pl-example04-jonne-doubled' '2 plates' '319'
	      'pl-example42' '1 plates' '336'
	      'pl-example20' '1 plates' '2221'
	      #'2020-09-30-jonne-slack' #'1 plates'
	    )

len=${#myUnitTests[@]}

for (( i=0; i<$len; i=i+3 ))
do
    echo "Testing file ${myUnitTests[${i}]}.dzn"
    echo "Expecting ${myUnitTests[${i}+1]}"
    echo "Usually takes about ${myUnitTests[${i}+2]} sec."

    SECONDS=0

    # Deterministic solution
#    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design.mzn ${myUnitTests[${i}]}.dzn --cmdline-data "testing=true"  &> ${myUnitTests[${i}]}.txt

    # Random and multi-thread
    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design.mzn regression-tests/${myUnitTests[${i}]}.dzn -p 10 -r $RANDOM --cmdline-data "testing=true"  &> ${myUnitTests[${i}]}.txt

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
    
#    if cmp "${myUnitTests[${i}]}.txt" "./output-regression-tests/${myUnitTests[${i}]}.txt";
 #   then
#	echo "The solution has not changed :-)"
 #   else
#	echo "WARNING: Check ${myUnitTests[${i}]}.dzn! Something changed!!!!!!!!\n"
 #   fi

    echo "\n"
done

echo "We're done now! Bye!"
