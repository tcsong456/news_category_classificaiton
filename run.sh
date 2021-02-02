echo 'running the program'
python train.py --cuda --batch_size_train 64 --batch_size_eval 64 \
--tokenizer treebank --vocab corpus/vocab_train.pkl --max_seq_len 64 \
--train_corpus corpus/corpus_train.txt --eval_corpus corpus/corpus_test.txt \
--mode cbow --bidirectional --epochs 15 --save_path corpus/news_clf_model.pth \
--use_word_embedding --embedding_trainable
