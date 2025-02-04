#!/bin/sh

echo "\nHello, world! Let's check that our PLAID constraint model v2 behaves as expected...\n"

## FileName N.Plates Time (in seconds)
myUnitTests=( 'pl-example-001' '1' '1'
	      'pl-example-002' '1' '1'
	      'pl-example-003' '1' '1' # Test empty corners
	      'pl-example-004' '1' '1' # Test odd number of rows
	      'pl-example-005' '1' '1' # Test odd number of controls
	      'pl-example-013' '1' '1' # Test old examples with new input format
	      'pl-example-014' '1' '1' # Test old examples
	      'pl-example-015' '1' '1' # Test old examples
	      'pl-example-017' '1' '1' # Test old examples
	      'pl-example-018' '1' '1' # Test old examples
	      'pl-example-019' '1' '1' # Test old examples
	      'pl-example-021' '1' '1' # Test old examples
	      'pl-example-042' '1' '1' # Test old examples
	    )

len=${#myUnitTests[@]}

for (( i=0; i<$len; i=i+3 ))
do
    echo "Testing file ${myUnitTests[${i}]}.dzn"
    
    echo "Expecting ${myUnitTests[${i+1}]} plates"
    
    echo "Usually takes about ${myUnitTests[${i}+2]} sec."

    SECONDS=0

    # Deterministic solution
    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design-main.mzn plate-design-control-constraints.mzn plate-design-basic-output.mzn dzn-examples/${myUnitTests[${i}]}.dzn --cmdline-data "testing=true"  &> regression-tests-results/${myUnitTests[${i}]}.txt

    # Random and multi-thread
#    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design-main.mzn plate-design-basic-output.mzn dzn-examples/${myUnitTests[${i}]}.dzn -p 10 -r $RANDOM --cmdline-data "testing=true"  &> regression-tests-results/${myUnitTests[${i}]}.txt

    echo "It took about $SECONDS sec."
    
    read -r result_line < regression-tests-results/${myUnitTests[${i}]}.txt
    
    set -- ${myUnitTests[${i}+1]} 

    
    # Check that we have the expected number of plates or expected error
    
    if [[ $result_line == "${myUnitTests[${i+1}]} plates" || $1 == "assert" ]]
    then
	echo "We are happy with the results of ${myUnitTests[${i}]}.txt :-)"
    else
	echo "ERROR: The number of plates in ${myUnitTests[${i}]}.dzn changed!!!!!!!!"
	break
    fi   

    echo "\n"
done

echo "We're done now! Bye!"
