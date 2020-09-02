#!/usr/bin/env bash

python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.0 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.1 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.2 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.3 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.4 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.5 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.6 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.7 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.8 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 0.9 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights dynamic_loss_size --weight_smoothing 1.0 --device cuda:4

python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights inverse --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights cost_sensitive --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights uniform --device cuda:4

python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.9999 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.999 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.99 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.9 --device cuda:4



python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.0 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.1 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.2 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.3 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.4 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.5 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.6 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.7 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.8 --device cuda:4
python main.py --data SemEval_Arabic_English_Spanish --experiment ablation_loss_plot2 --batch_size 32 --dropout 0.22014317130885194 --gamma 4.0 --hidden_size 128 --loss focal --lr 0.0010100156584368527 --prepro all --thresholding class_specific --num_layers 2 --weight_decay 1.4546921895462235e-08 --weights class_balanced --beta 0.9 --device cuda:4
wait