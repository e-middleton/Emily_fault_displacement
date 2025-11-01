#! /bin/bash

fileNames=("./_outputs/D40_SU_Testing/test0" ) # "./_outputs/D40_SU_Testing/test1" "./_outputs/D40_SU_Testing/test2" "./_outputs/D40_SU_Testing/test3" "./_outputs/D40_SU_Testing/test4") #"./_outputs/D40_SU_Testing/test1")

i=0

for name in ${fileNames[@]}; do

    python3 main.py "--saveFigures" "--oldResults" "--resultFile=$name"
    mv *.pdf ./_outputs/D40_SU_Testing/test$i/
    # mv configSettings.txt ./_outputs/D40_SU_Testing/test$i/

    i=$((i+1)) # update testFile
    echo # empty line
        
done