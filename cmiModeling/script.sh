#! /bin/bash


saveFigs="True"
saveData="True"
testFile="test5"
testName="--testName=$testFile"

# check if test file directory exists
if [ -d "./_outputs/$testFile" ] ; then 
echo "test directory already exists, not recreating"
else
echo "creating test directory"
cd _outputs
mkdir $testFile
cd ..
fi # end the if-else statement

if [ "$saveFigs" == "True" ] && [ "$saveData" == "True" ] # run the model, and save figures if that tag is set
then
    echo "Save figures is true"
    echo "Save data is true"
    python3 main.py "--saveFigures" "--saveData" $testName
    mv *.pdf ./_outputs/$testFile/
    mv numericalResults.txt ./_outputs/$testFile/
    mv configSettings.txt ./_outputs/$testFile/

elif [ "$saveFigs" == "True" ]; then
    echo "save figures is true"
    python3 main.py "--saveFigures" $testName
    mv *.pdf ./_outputs/$testFile/
    mv configSettings.txt ./_outputs/$testFile/

elif [ "saveData" == "True" ]; then
    echo "save data is true"
    python3 main.py "--saveData" $testName
    mv numericalResults.txt ./_outputs/$testFile/
    mv configSettings.txt ./_outputs/$testFile/
else
    python3 main.py $testName
fi

