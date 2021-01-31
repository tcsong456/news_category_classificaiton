echo 'builing vocab'
if [ ! -d "fastText" ];then
  git clone https://github.com/facebookresearch/fastText.git
  cd fastText
  pip install .
  cd ..
fi
awk -F '\t' '{print tolower($2)}' corpus/corpus_train.txt > corpus/corpus_train_text.txt
python build_vocab.py --tokenizer 'treebank' --max_seq_len 1024 --input \
'corpus/corpus_train_text.txt' --corpus 'corpus/corpus_train.txt' --vocab \
'corpus/vocab_train.pkl' --min_freq 5 --mode 'skipgram' --lower \
--pretrained_vectors
