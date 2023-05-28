set -o pipefail

python src/offline/summary_generation.py --split=train --model-name=t5-base --batch-size=50 --limit=1000
python src/offline/summary_generation.py --split=valid --model-name=t5-base --batch-size=50 --limit=100

python src/offline/summary_classification_length.py

python src/train/train_slic.py --model-name=t5-base --batch-size=6 --limit=1000 --loss-type=slic_loss 
python src/train/train_slic.py --model-name=t5-base --batch-size=6 --limit=1000 --loss-type=slic_loss_logits 
