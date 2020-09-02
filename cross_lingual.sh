ARABIC_BEST=28b66120527f4530826c9b2b78c8ded3
ARABIC_DEVICE=cuda:2
echo "With Best Arabic..."
python main.py --experiment cross_lingual_val --no_train --finetune $ARABIC_BEST --data SemEval_Arabic --device $ARABIC_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $ARABIC_BEST --data SemEval_English --device $ARABIC_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $ARABIC_BEST --data SemEval_Spanish --device $ARABIC_DEVICE

ENGLISH_BEST=9b8dbce5a76647cb8d4eb83a30fdac44
ENGLISH_DEVICE=cuda:1
echo "With Best English..."
python main.py --experiment cross_lingual_val --no_train --finetune $ENGLISH_BEST --data SemEval_Arabic --device $ENGLISH_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $ENGLISH_BEST --data SemEval_English --device $ENGLISH_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $ENGLISH_BEST --data SemEval_Spanish --device $ENGLISH_DEVICE

SPANISH_BEST=a54141fa0d5e41c280e2a1bfdf8699f3
SPANISH_DEVICE=cuda:3
echo "With Best Spanish..."
python main.py --experiment cross_lingual_val --no_train --finetune $SPANISH_BEST --data SemEval_Arabic --device $SPANISH_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $SPANISH_BEST --data SemEval_English --device $SPANISH_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $SPANISH_BEST --data SemEval_Spanish --device $SPANISH_DEVICE

AR_EN_BEST=0e418553aa1d44628feb2a5d49442f2d
AR_EN_DEVICE=cuda:0
echo "With Best AR+EN..."
python main.py --experiment cross_lingual_val --no_train --finetune $AR_EN_BEST --data SemEval_Arabic --device $AR_EN_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $AR_EN_BEST --data SemEval_English --device $AR_EN_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $AR_EN_BEST --data SemEval_Spanish --device $AR_EN_DEVICE

AR_SP_BEST=b48ef1fbb4e0412ba8e64c5523257982
AR_SP_DEVICE=cuda:1
echo "With Best AR+SP..."
python main.py --experiment cross_lingual_val --no_train --finetune $AR_SP_BEST --data SemEval_Arabic --device $AR_SP_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $AR_SP_BEST --data SemEval_English --device $AR_SP_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $AR_SP_BEST --data SemEval_Spanish --device $AR_SP_DEVICE

EN_SP_BEST=bd82b8599e1f49d5b60671b3789a997d
EN_SP_DEVICE=cuda:2
echo "With Best EN+SP..."
python main.py --experiment cross_lingual_val --no_train --finetune $EN_SP_BEST --data SemEval_Arabic --device $EN_SP_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $EN_SP_BEST --data SemEval_English --device $EN_SP_DEVICE
python main.py --experiment cross_lingual_val --no_train --finetune $EN_SP_BEST --data SemEval_Spanish --device $EN_SP_DEVICE
