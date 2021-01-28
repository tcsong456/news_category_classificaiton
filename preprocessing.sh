echo 'cleaning text'
python preprocessing.py --corpus corpus/corpus.txt > corpus/corpus_clean.txt
head -n 5 corpus/corpus_clean.txt
