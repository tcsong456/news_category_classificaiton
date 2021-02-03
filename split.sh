echo 'splitting train test'
if [ $# -ne 2 ];then
  echo 'train test split percentage must be provided!'
  exit 1
else
  total=$(echo "$1 + $2" | bc)
  if [ "$(echo "$total != 1.0" | bc)" -eq 1 ];then
    echo 'the sum of train and test ratio must equal to 1'
    exit 1
  elif [ "$(echo "$1 < 0" | bc)" -eq 1 ] || [ "$(echo "$1 > 1" | bc)" -eq 1 ];then
    echo 'train ratio must be between 0 and 1'
    exit 1
  elif [ "$(echo "$2 < 0" | bc)" -eq 1 ] || [ "$(echo "$2 > 1" | bc)" -eq 1 ];then
    echo 'test ratio must be between 0 and 1'
    exit 1
  else
    lines=$(wc -l < corpus/corpus_clean.txt)
    train_lines=$(echo "scale=0; $1 * $lines" | bc)
    train_lines=$(echo $train_lines | cut -d'.' -f1)
    test_lines=$(echo "scale=0; $lines - $train_lines" | bc)
    cat corpus/corpus_clean.txt | shuf > corpus/corpus_shuf.txt
    head -n $train_lines corpus/corpus_shuf.txt > corpus/corpus_train.txt
    tail -n $test_lines corpus/corpus_shuf.txt > corpus/corpus_test.txt
    wc -l corpus/corpus_train.txt corpus/corpus_test.txt
  fi
fi
