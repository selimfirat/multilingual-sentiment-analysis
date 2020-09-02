#!/usr/bin/env bash
for DATA in SE0714 Olympic SS-Twitter SS-Youtube SCv1 SCv2-GEN PsychExp SemEval
do
    FIRST=true
    for LR in 0.1 0.01 0.001 0.0001
    do
        for WNG in 1 2 3 4 5 6
        do
            for bucket in 500000 1000000 2000000 4000000 8000000
            do
                for neg in 1 2 3 4 5 6
                do
                    for loss in ns hs softmax
                    do
                        for dim in 25 50 100 200 400
                        do
                            for ws in 1 2 3 4 5 6
                            do
                                for t in 0.1 0.01 0.001 0.0001 0.00001
                                do
                                    echo "--lr {$LR} --wordNgrams {$WNG} --bucket {$bucket} --neg {$neg} --loss {$loss} --dim {$dim} --ws {$ws} --t {$t} --num_epochs=100"
                                    python main.py --module fasttext --data $DATA --lr $LR --wordNgrams $WNG --bucket $bucket --neg $neg --loss $loss --dim $dim --ws $ws --t $t --num_epochs=100
                                    if $FIRST
                                    then
                                        wait
                                        FIRST=false
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
        wait
    done
done