#!/bin/bash
git submodule update --init --recursive

conda create -n new-hadpo python=3.10 -y
conda activate new-hadpo

pip install -r requirements.txt
python -m nltk.downloader all
python -m spacy download en_core_web_lg
gdown https://drive.google.com/uc?id=1MaCHgtupcZUjf007anNl4_MV0o4DjXvl -O AMBER/data/AMBER.zip
cd AMBER/data
unzip AMBER.zip
rm -f AMBER.zip
cd ../..