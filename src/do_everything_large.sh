python src/offline/summary_generation.py --split=train --model-name=t5-large
python src/offline/summary_generation.py --split=valid --model-name=t5-large

python src/offline/summary_classification_length.py

python src/train/train_slic.py --loss-type=slic_loss --model-name=t5-large
python src/train/train_slic.py --loss-type=slic_loss_logits --model-name=t5-large
