#!/bin/bash

. ./path.sh
. ./cmd.sh

nj=50
ofmt="xx"
k=3
sil_symb="@"
rep_factor=1

. ./utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: ./get_sil_durs.sh <data> <lang> <src> <ali> <sil_trans>"
  exit 1;
fi

data=$1
lang=$2
exp=$3
odir=$4
sil_trans=$5

mkdir -p ${odir}
sil_phones=$(grep 'SIL' ${lang}/phones.txt | awk '{print $2}' | sort | awk '{printf("%s", $1)}')
sil_phoneB=${sil_phones:0:1}
sil_phoneE=${sil_phones: -1}
#./steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj $data $lang $exp $odir
ali-to-phones --write-lengths \
  $exp/final.mdl ark:"gunzip -c ${odir}/ali.*.gz |" ark,t:- |\
  sed "s/; [${sil_phones}] /; SIL /g" |\
  awk -v silB=${sil_phoneB} -v silE=${sil_phoneE} '{ 
    printf("%s ", $1);
    if($2 <= silE && $2 >= silB) {
      printf("SIL ")
    } 
    else {
      printf("%s ", $2)
    }
    for (i=3; i<=NF; i++) {
      printf("%s ", $i)
    }
    printf("\n")
  }' > ${odir}/sil_durs


python sil_durs_to_sil_trans.py --ofmt $ofmt --sil-symb ${sil_symb} --num-clusters $k --rep-factor $rep_factor\
  $sil_trans ${odir}/sil_durs

