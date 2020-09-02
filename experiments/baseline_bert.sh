#!/usr/bin/env bash
for DATA in SE0714 Olympic SS-Twitter SS-Youtube SCv1 SCv2-GEN PsychExp SemEval
do
    FIRST=true
    for LR in 0.1 0.01 0.001 0.0001
    do
        for WD in 1e-5 1e-6 1e-7 0
        do
            python main.py --module bert --device cuda:1 --experiment baseline_bert_$DATA --data $DATA --lr $LR --weight_decay $WD --freeze_bert False
            if $FIRST
            then
                wait
                FIRST=false
            fi
        done
        wait
    done
done