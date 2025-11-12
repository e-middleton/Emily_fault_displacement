#! /bin/bash

# uniform smoothing models
cd _outputs2
mkdir D40_SU_Testing
mkdir D50_SU_Testing
mkdir D60_SU_Testing
mkdir D80_SU_Testing
mkdir D40_SV_Testing
mkdir D50_SV_Testing
mkdir D60_SV_Testing
mkdir D80_SV_Testing
cd ..

# 40km SU models

faultWeight=(50000000000000 50000000000000 50000000000000 100000000000000 100000000000000 100000000000000 400000000000000)
cmiWeight=(50000000000000 100000000000000 400000000000000 100000000000000 400000000000000 1000000000000000 1000000000000000)

i=0 #loop var
for model in ${faultWeight[@]}; do
    if [ -d "./_outputs2/test$i" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test$i"
        cd ..
    fi
    cmi=${cmiWeight[$i]}
    python3 main.py "--saveFigures" "--saveData" "--planeDepth=40" "--testName=test$i" "--faultSmoothing=$model" "--cmiSmoothing=$cmi" "gpsFile=./gpsDiao/gpsDiao1.txt"
    mv *.pdf ./_outputs2/test$i/
    mv numericalResults.txt ./_outputs2/test$i/
    mv configSettings.txt ./_outputs2/test$i/
    mv *.npy ./_outputs2/test$i/

    i=$((i+1))
    echo # empty line
    
done

cd _outputs2
mv test* ./D40_SU_Testing
cd ..

# 50km SU models

faultWeight2=(50000000000000 100000000000000 100000000000000 400000000000000 400000000000000)
cmiWeight2=(1000000000000000 400000000000000 1000000000000000 400000000000000 1000000000000000)

m=0 #loop var
for model in ${faultWeight2[@]}; do
    if [ -d "./_outputs2/test$m" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test$m"
        cd ..
    fi
    cmi=${cmiWeight2[$m]}
    python3 main.py "--saveFigures" "--saveData" "--planeDepth=50" "--testName=test$m" "--faultSmoothing=$model" "--cmiSmoothing=$cmi" "gpsFile=./gpsDiao/gpsDiao1.txt"
    mv *.pdf ./_outputs2/test$m/
    mv numericalResults.txt ./_outputs2/test$m/
    mv configSettings.txt ./_outputs2/test$m/
    mv *.npy ./_outputs2/test$m/

    m=$((m+1))
    echo # empty line
    
done

cd _outputs2
mv test* ./D50_SU_Testing
cd ..

#60km SU models
faultWeight3=(50000000000000 100000000000000 100000000000000 400000000000000 400000000000000)
cmiWeight3=(1000000000000000 400000000000000 1000000000000000 400000000000000 1000000000000000)

n=0 #loop var
for model in ${faultWeight3[@]}; do
    if [ -d "./_outputs2/test$n" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test$n"
        cd ..
    fi
    cmi=${cmiWeight3[$n]}
    python3 main.py "--saveFigures" "--saveData" "--planeDepth=60" "--testName=test$n" "--faultSmoothing=$model" "--cmiSmoothing=$cmi" "gpsFile=./gpsDiao/gpsDiao1.txt"
    mv *.pdf ./_outputs2/test$n/
    mv numericalResults.txt ./_outputs2/test$n/
    mv configSettings.txt ./_outputs2/test$n/
    mv *.npy ./_outputs2/test$n/

    n=$((n+1))
    echo # empty line
    
done

cd _outputs2
mv test* ./D60_SU_Testing
cd ..

#80km SU models
faultWeight4=(1000000000000000)
cmiWeight4=(1000000000000000)

if [ -d "./_outputs2/test0" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test0"
        cd ..

python3 main.py "--saveFigures" "--saveData" "--planeDepth=80" "--testName=test0" "--faultSmoothing=1000000000000000" "--cmiSmoothing=1000000000000000" "gpsFile=./gpsDiao/gpsDiao1.txt"

mv *.pdf ./_outputs2/test0
mv numericalResults.txt ./_outputs2/test0/
mv configSettings.txt ./_outputs2/test0/
mv *.npy ./_outputs2/test0/

cd _outputs2
mv test0 ./D80_SU_Testing
cd ..

# spatially variable models

#40km SV models
faultWeight5=(10000000000000 10000000000000 100000000000000 100000000000000 100000000000000 1000000000000000)
cmiWeight5=(50000000000000 100000000000000 10000000000000 50000000000000 100000000000000 400000000000000 10000000000000)

p=0 #loop var
for model in ${faultWeight5[@]}; do
    if [ -d "./_outputs2/test$p" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test$p"
        cd ..
    fi
    cmi=${cmiWeight5[$p]}
    python3 main.py "--saveFigures" "--saveData" "--planeDepth=40" "--testName=test$p" "--faultSmoothing=$model" "--cmiSmoothing=$cmi" "gpsFile=./gpsDiao/gpsDiao1.txt" "--spatiallyVariable"
    mv *.pdf ./_outputs2/test$p/
    mv numericalResults.txt ./_outputs2/test$p/
    mv configSettings.txt ./_outputs2/test$p/
    mv *.npy ./_outputs2/test$p/

    p=$((p+1))
    echo # empty line
    
done

cd _outputs2
mv test* ./D40_SV_Testing
cd ..

#50km SV models

faultWeight6=(1000000000000 1000000000000 5000000000000 5000000000000 5000000000000 5000000000000 10000000000000 10000000000000 10000000000000)
cmiWeight6=(50000000000000 100000000000000 50000000000000 100000000000000 400000000000000 1000000000000000 100000000000000 400000000000000 1000000000000000)

l=0 #loop var
for model in ${faultWeight6[@]}; do
    if [ -d "./_outputs2/test$l" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test$l"
        cd ..
    fi
    cmi=${cmiWeight6[$l]}
    python3 main.py "--saveFigures" "--saveData" "--planeDepth=50" "--testName=test$l" "--faultSmoothing=$model" "--cmiSmoothing=$cmi" "gpsFile=./gpsDiao/gpsDiao1.txt" "--spatiallyVariable"
    mv *.pdf ./_outputs2/test$l/
    mv numericalResults.txt ./_outputs2/test$l/
    mv configSettings.txt ./_outputs2/test$l/
    mv *.npy ./_outputs2/test$l/

    l=$((l+1))
    echo # empty line
    
done

cd _outputs2
mv test* ./D50_SV_Testing
cd ..

#60km SV models

faultWeight7=(5000000000000 5000000000000 5000000000000 5000000000000 5000000000000 10000000000000 10000000000000 10000000000000 10000000000000)
cmiWeight7=(10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)

q=0 #loop var
for model in ${faultWeight7[@]}; do
    if [ -d "./_outputs2/test$q" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test$q"
        cd ..
    fi
    cmi=${cmiWeight7[$q]}
    python3 main.py "--saveFigures" "--saveData" "--planeDepth=40" "--testName=test$q" "--faultSmoothing=$model" "--cmiSmoothing=$cmi" "gpsFile=./gpsDiao/gpsDiao1.txt" "--spatiallyVariable"
    mv *.pdf ./_outputs2/test$q/
    mv numericalResults.txt ./_outputs2/test$q/
    mv configSettings.txt ./_outputs2/test$q/
    mv *.npy ./_outputs2/test$q/

    q=$((q+1))
    echo # empty line
    
done

cd _outputs2
mv test* ./D60_SV_Testing
cd ..

#80km SV models
faultWeight8=(1000000000000 5000000000000 5000000000000 5000000000000 5000000000000 10000000000000 10000000000000 10000000000000 10000000000000 10000000000000)
cmiWeight8=(100000000000000 5000000000000 10000000000000 50000000000000 100000000000000 10000000000000 50000000000000 100000000000000 400000000000000 1000000000000000)

r=0 #loop var
for model in ${faultWeight8[@]}; do
    if [ -d "./_outputs2/test$r" ] ; then 
        echo "test directory already exists, not recreating"
    else
        echo "creating test directory"
        cd _outputs2
        mkdir "test$r"
        cd ..
    fi
    cmi=${cmiWeight8[$r]}
    python3 main.py "--saveFigures" "--saveData" "--planeDepth=80" "--testName=test$r" "--faultSmoothing=$model" "--cmiSmoothing=$cmi" "gpsFile=./gpsDiao/gpsDiao1.txt" "--spatiallyVariable"
    mv *.pdf ./_outputs2/test$r/
    mv numericalResults.txt ./_outputs2/test$r/
    mv configSettings.txt ./_outputs2/test$r/
    mv *.npy ./_outputs2/test$r/

    r=$((r+1))
    echo # empty line
    
done

cd _outputs2
mv test* ./D80_SV_Testing
cd ..

