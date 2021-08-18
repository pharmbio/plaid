#!/bin/sh
 
echo "Hello, world!"

## Remember to set testing = true

myUnitTests=( 'pl-example01'
	      'pl-example02'
	      'pl-example03'
	      'pl-example04-jonne-doubled'
	      'pl-example05'
	      'pl-example06' 'pl-example07-tiny' 'pl-example08-small' 'pl-example09' '2020-11-13-jonne-slack'
	      'pl-example10'
	      'pl-example11' 'pl-example12' 'pl-example13'
	      'pl-example14'
	      'pl-example15' 'pl-example17' 'pl-example18' 'pl-example19'
	      '2020-10-08-jonne-slack' 'pl-example21' 'pl-example22' 'pl-example23' 'pl-example24'
	      'pl-example25' 'pl-example27' 'pl-example28' 'pl-example29' 'pl-example30' 'pl-example35'
	      'pl-example20' #'2020-09-30-jonne-slack'
	    )

myUnitTestsResults=( '4 plates'
		     'MiniZinc: evaluation error:'
		     'MiniZinc: evaluation error:'
		     '2 plates'
		     '2 plates'  #'pl-example05'
		     '2 plates' '4 plates' '2 plates' '2 plates' '1 plates'
		     '4 plates'
		     '2 plates' '2 plates' '1 plates'
		     '1 plates' #'pl-example14'
		     '1 plates' '1 plates' '1 plates' '1 plates'
		     '1 plates' '1 plates' '4 plates' '2 plates' '2 plates' '2 plates' '3 plates' '4 plates'
		     '2 plates' '1 plates'
		     '1 plates' #'pl-example35'
		     '1 plates' #'pl-example20'
		     #'1 plates' #'2020-09-30-jonne-slack'
		   )

len=${#myUnitTests[@]}

for (( i=0; i<$len; i++ ))
do
    echo "Testing file ${myUnitTests[${i}]}.dzn"
    echo "Expecting ${myUnitTestsResults[${i}]}"

    SECONDS=0
    
    /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode plate-design.mzn ${myUnitTests[${i}]}.dzn --cmdline-data "testing=true"  &> ${myUnitTests[${i}]}.txt
    
     #/Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver Gecode -p 8 -t 7200000 -r $RANDOM plate-design.mzn ${myUnitTests[${i}]}.dzn  --cmdline-data testing=true > ${myUnitTests[${i}]}.txt

    echo "It took about $SECONDS sec."
    
    read -r line < ${myUnitTests[${i}]}.txt
    
    set -- ${myUnitTestsResults[${i}]} 

    
    # Check that we have the expected number of plates or expected error
    
    if [[ $line == ${myUnitTestsResults[${i}]} || $1 == "assert" ]]
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
