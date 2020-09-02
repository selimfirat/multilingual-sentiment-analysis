python main.py --experiment sota_val --finetune 2178c82b9bc249cea3b485a83c3ecb5c --no_train --data SemEval_Arabic
python main.py --experiment sota_val --finetune 2178c82b9bc249cea3b485a83c3ecb5c --no_train --data SemEval_English
python main.py --experiment sota_val --finetune 2178c82b9bc249cea3b485a83c3ecb5c --no_train --data SemEval_Spanish

python main.py --experiment sota_ours_sl --prepro all --dropout 0.22014317130885194 --lr 0.0010100156584368527 --num_layers 2 --hidden_size 128 --weight_decay 1.4546921895462235e-08 --weight_smoothing 0.2965246785991912 --gamma 4.0 --batch_size 32 --data SemEval_Arabic
python main.py --experiment sota_ours_sl --prepro all --dropout 0.22014317130885194 --lr 0.0010100156584368527 --num_layers 2 --hidden_size 128 --weight_decay 1.4546921895462235e-08 --weight_smoothing 0.2965246785991912 --gamma 4.0 --batch_size 32 --data SemEval_English
python main.py --experiment sota_ours_sl --prepro all --dropout 0.22014317130885194 --lr 0.0010100156584368527 --num_layers 2 --hidden_size 128 --weight_decay 1.4546921895462235e-08 --weight_smoothing 0.2965246785991912 --gamma 4.0 --batch_size 32 --data SemEval_Spanish

python main.py --finetune 2178c82b9bc249cea3b485a83c3ecb5c --data PsychExp --finetune_type concat_final_freezed
python main.py --finetune 2178c82b9bc249cea3b485a83c3ecb5c --data Olympic --finetune_type concat_final_freezed
python main.py --finetune 2178c82b9bc249cea3b485a83c3ecb5c --data SE0714 --finetune_type concat_final_freezed
