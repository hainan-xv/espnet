#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false
bpe=false
remove_blank=true
filter=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn.py ${dir}/data.json ${dic} ${dir}/ref.trn ${dir}/hyp.trn

if $bpe; then
    sed -i.bak1 -r 's/(@@ )|(@@ ?$)//g' ${dir}/ref.trn
    sed -i.bak1 -r 's/(@@ )|(@@ ?$)//g' ${dir}/hyp.trn
fi
if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
fi

for i in hyp ref; do
  mv $dir/$i.trn $dir/$i.trn.tri
  cat $dir/$i.trn.tri | awk '{print $NF}' > $dir/tmp.uttid.$i
  cat $dir/$i.trn.tri | awk '{for(i=1;i<NF;i++) printf("%s ",$i); print""}' | sed "s=-= =g" | awk '{for(i=2;i<=NF;i+=3) printf("%s ",$i); print""}' > $dir/tmp.content.$i
  paste $dir/tmp.content.$i $dir/tmp.uttid.$i > $dir/$i.trn
done
    
sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt

    echo "write a WER result in ${dir}/result.wrd.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
