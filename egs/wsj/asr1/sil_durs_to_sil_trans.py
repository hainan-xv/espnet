from __future__ import print_function
import argparse
import sys
from sklearn.cluster import KMeans
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sil_trans', help="transcription with silence",
        type=str)
    parser.add_argument('sil_durs', help="ali-to-phones output", type=str)
    parser.add_argument('--sil-symb', default='@', action='store', type=str)
    parser.add_argument('--ofmt', default='xx', choices=['xx', 'x2'], action='store', type=str)
    parser.add_argument('--num-clusters', default=3, action='store', type=int)
    parser.add_argument('--rep-factor', default=1, action='store', type=float)
    args = parser.parse_args()

    with open(args.sil_durs, 'r') as f:
        lines = f.readlines()

    # Get silence durations
    lines2 = {}
    for l in lines:
        utt_id, utt = l.strip().split(None, 1)
        lines2[utt_id] = [int(p.split()[1]) for p in utt.split(' ; ') if p.split()[0] == 'SIL']


    # Kmean "binning"
    if args.ofmt == "x2":
        sils = []
        for utt, sil_frames in lines2.iteritems():
            sils.extend(sil_frames)

        sils = np.array([sils]).T
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
        kmeans.fit(sils)

    # Sil replace functions
    def _fmt_sil_xx(symb, count):
        return ' '.join(symb for i in range(int(round(args.rep_factor * count))))

    def _fmt_sil_x2(symb, count): 
        return symb + str(kmeans.predict([[count]])[0])

    # Select sil replace function based on output format
    if args.ofmt == "x2":
        sil_fun = _fmt_sil_x2
    else:
        sil_fun = _fmt_sil_xx

    # Parse sil transcription file
    with open(args.sil_trans, 'r') as f:
        for l in f:
            utt_id, utt = l.strip().split(None, 1)
            utt_new = ""
            words = utt.split()
            assert words.count(args.sil_symb) == len(lines2[utt_id]), \
            "Number of sil does not match: "
            "utterance {}: {} v. {}".format(utt_id, words.count(args.sil_symb),
                                            len(lines2[utt_id])) 
            for p in words:
                sil_count = 0
                if p == args.sil_symb:
                    new_sil = sil_fun(args.sil_symb, lines2[utt_id][sil_count]) 
                    if new_sil.strip() != '':
                        utt_new += new_sil + " "
                    sil_count += 1
                else:
                    utt_new += p + " "  
            print("{} {}".format(utt_id, utt_new)) 


if __name__ == "__main__":
    main()

