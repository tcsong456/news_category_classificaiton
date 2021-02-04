echo 'splitting train test'
if [ $# -ne 2 ];then
  echo 'both train and valid split ratio must be provided'
  exit 1
else
  total=$(echo $1+$2 | bc)
  if [ "$(echo "$total != 1.0" | bc)" -eq 1 ];then
    echo 'the sum of train and valid ratio must be one'
  elif [ "$(echo "$1 < 0" | bc)" -eq 1 ] or [ "$(echo "$1 > 1" | bc)" -eq 1 ];then
    echo 'the first parameter must be between 0 and 1'
  elif [ "$(echo "$2 < 0" | bc)" -eq 1 ] or [ "$(echo "$2 > 1" | bc)" -eq 1 ];then
    echo 'the second parameter must be between 0 and 1'
  else
    lines=$(wl -l < corpus_clean.txt)
    train_lines=$(echo "scale=0; $1*$lines" | bc)
    train_lines=$(echo $train_lines | awk -F '.' '{print $1}')
    valid_liens=$(echo "scale=0; $lines-$train_lines" | bc)
    cat corpus_clean.txt | shuf > corput_shuf.txt
    head -n $train_lines corpus_shuf.txt > corpus_train.txt
    tail -n $valid_lines corpus_shuf.txt > corpus_valid.txt
  fi
fi
