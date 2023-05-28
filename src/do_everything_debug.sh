python src/offline/summary_generation.py --split=train --debug
python src/offline/summary_generation.py --split=valid --debug

python src/offline/summary_classification_length.py

python src/train/train_slic.py --loss-type=slic_loss --learning-rate=1e-4 --debug
python src/train/train_slic.py --loss-type=slic_loss_logits --learning-rate=1e-4 --debug
