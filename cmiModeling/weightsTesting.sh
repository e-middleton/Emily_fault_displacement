#! /bin/bash

cd _outputs
mkdir D40_SU_Testing
mkdir D50_SU_Testing
mkdir D40_SV_Testing
mkdir D50_SV_Testing
cd ..


# run the D40 SU testing with weights 1e13-1e15
faultWeights1=(10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)
cmiWeights1=(10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)

i=0 #loop variable


for fault in ${faultWeights1[@]}; do 
    for cmi in ${cmiWeights1[@]}; do

        if [ -d "./_outputs/test$i" ] ; then 
            echo "test directory already exists, not recreating"
        else
            echo "creating test directory"
            cd _outputs
            mkdir "test$i"
            cd ..
        fi # end the if-else statement

        python3 main.py "--saveFigures" "--saveData" "--planeDepth=40" "--testName=test$i" "--faultSmoothing=$fault" "--cmiSmoothing=$cmi"
        mv *.pdf ./_outputs/test$i/
        mv numericalResults.txt ./_outputs/test$i/
        mv configSettings.txt ./_outputs/test$i/
        mv *.npy ./_outputs/test$i/

        i=$((i+1))
        echo # empty line
        
    done
done

cd _outputs
mv test* ./D40_SU_Testing
cd ..

### run D50 SU Testing ###

faultWeights2=(10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)
cmiWeights2=(10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)

m=0 #loop variable

for fault in ${faultWeights2[@]}; do 
    for cmi in ${cmiWeights2[@]}; do

        if [ -d "./_outputs/test$m" ] ; then 
            echo "test directory already exists, not recreating"
        else
            echo "creating test directory"
            cd _outputs
            mkdir "test$m"
            cd ..
        fi # end the if-else statement

        python3 main.py "--saveFigures" "--saveData" "--planeDepth=50" "--testName=test$m" "--faultSmoothing=$fault" "--cmiSmoothing=$cmi"
        mv *.pdf ./_outputs/test$m/
        mv numericalResults.txt ./_outputs/test$m/
        mv configSettings.txt ./_outputs/test$m/
        mv *.npy ./_outputs/test$m/

        m=$((m+1))
        echo # empty line
        
    done
done

cd _outputs
mv test* ./D50_SU_Testing
cd ..

# run D40 SV Testing
faultWeights3=(1000000000000 5000000000000 10000000000000 50000000000000 100000000000000)
cmiWeights3=(1000000000000 5000000000000 10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)

l=0 #loop variable

for fault in ${faultWeights3[@]}; do 
    for cmi in ${cmiWeights3[@]}; do

        if [ -d "./_outputs/test$l" ] ; then 
            echo "test directory already exists, not recreating"
        else
            echo "creating test directory"
            cd _outputs
            mkdir "test$l"
            cd ..
        fi # end the if-else statement

        python3 main.py "--saveFigures" "--saveData" "--planeDepth=40" "--spatiallyVariable" "--testName=test$l" "--faultSmoothing=$fault" "--cmiSmoothing=$cmi"
        mv *.pdf ./_outputs/test$l/
        mv numericalResults.txt ./_outputs/test$l/
        mv configSettings.txt ./_outputs/test$l/
        mv *.npy ./_outputs/test$l/

        l=$((l+1))
        echo # empty line
        
    done
done

cd _outputs
mv test* ./D40_SV_Testing
cd ..

# run D50 SV Testing

faultWeights4=(1000000000000 5000000000000 10000000000000 50000000000000 100000000000000)
cmiWeights4=(1000000000000 5000000000000 10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)

p=0 #loop variable

for fault in ${faultWeights4[@]}; do 
    for cmi in ${cmiWeights4[@]}; do

        if [ -d "./_outputs/test$p" ] ; then 
            echo "test directory already exists, not recreating"
        else
            echo "creating test directory"
            cd _outputs
            mkdir "test$p"
            cd ..
        fi # end the if-else statement

        python3 main.py "--saveFigures" "--saveData" "--spatiallyVariable" "--planeDepth=50" "--testName=test$p" "--faultSmoothing=$fault" "--cmiSmoothing=$cmi"
        mv *.pdf ./_outputs/test$p/
        mv numericalResults.txt ./_outputs/test$p/
        mv configSettings.txt ./_outputs/test$p/
        mv *.npy ./_outputs/test$p/

        p=$((p+1))
        echo # empty line
        
    done
done

cd _outputs
mv test* ./D50_SV_Testing
cd ..