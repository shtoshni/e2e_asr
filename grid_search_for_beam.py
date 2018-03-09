from __future__ import print_function

from os import path

import argparse
import subprocess
import sys


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cmd_file", type=str,
                        help="Command file to run the model")
    parser.add_argument("-use_lm", default=False, action="store_true",
                        help="Use LM in decoding")
    args = parser.parse_args()
    return args


def read_command(cmd_file):
    """Command file"""
    with open(cmd_file) as cmd_f:
        cmd = cmd_f.readline().strip()
        return cmd

def parse_output(output_str):
    out_file_pattern = "Output at:"
    score_pattern = "Score: "
    for out_line in output_str.splitlines():
        if out_file_pattern in out_line:
            out_file = out_line.split(out_file_pattern)[1]
        elif score_pattern in out_line:
            score = float(out_line.split(score_pattern)[1])
    return score, out_file


def grid_search(args):
    """Perform grid search on beam configurations and run the best config on test set."""
    base_cmd = read_command(args.cmd_file)
    cmd_dir = path.dirname(args.cmd_file)

    perf_file = path.join(cmd_dir, "perf.txt")

    dev_cmd = base_cmd + " -eval_dev "
    # Store best performances
    best_asr_perf = 1.00
    best_beam_size = 1
    best_lm_weight = 0

    if args.use_lm:
        lm_weight_options = [0, 0.25, 0.5, 0.75]
    else:
        lm_weight_options = [0]

    perf_dict = {}
    if path.isfile(perf_file):
        with open(perf_file, "r") as perf_f:
            for line in perf_f.readlines():
                beam_size, lm_weight, asr_perf = line.strip().split()
                # Prepare the keys
                beam_size = int(beam_size)
                lm_weight = round(float(lm_weight), 2)
                perf_dict[(beam_size, lm_weight)] = float(asr_perf)
        print ("Loaded %d entries from grid search" %(len(perf_dict)))
        sys.stdout.flush()

    with open(perf_file, "a", 0) as perf_f:
        for beam_size in [2, 4, 8, 16]:
        #for beam_size in [2]:
            print ("\nBeam size: %d" %beam_size)
            sys.stdout.flush()

            beam_best_perf = 1.0
            for lm_weight in lm_weight_options:
                query_key = (beam_size, round(lm_weight, 2))
                if perf_dict.has_key(query_key):
                    print ("From previous exec: ", end="")
                    asr_perf = perf_dict[query_key]
                else:
                    exec_cmd = (dev_cmd + " -beam_size " + str(beam_size)
                                + " -lm_weight " + str(lm_weight))
                    output = subprocess.check_output(exec_cmd, shell=True)
                    asr_perf, _= parse_output(output)
                    perf_f.write("%d %.2f %f\n" %(beam_size, lm_weight, asr_perf))
                    perf_f.flush()

                print ("ASR Error: %.4f, Beam size: %d, lm weight: %.2f" %
                       (asr_perf, beam_size, lm_weight))
                sys.stdout.flush()
                if beam_best_perf > asr_perf:
                    beam_best_perf = asr_perf
                else:
                    # The performance for a given beam size can only go downhill
                    # by increasing lm_weight further
                    print ("Not exploring further increasing lm_weight")
                    sys.stdout.flush()
                    break

                if best_asr_perf > asr_perf:
                    print ("Best config updated!!")
                    sys.stdout.flush()
                    best_asr_perf = asr_perf
                    best_beam_size = beam_size
                    best_lm_weight = lm_weight

    test_cmd = (base_cmd + " -test " + " -beam_size " + str(best_beam_size) +
                " -lm_weight " + str(best_lm_weight))
    output = subprocess.check_output(test_cmd, shell=True)
    _, out_file = parse_output(output)

    # Run the score.sh command finally
    out_dir = path.join(cmd_dir, "final_eval")

    subprocess.call("mkdir -p " + out_dir, shell=True)

    score_script_loc = "/share/data/speech/shtoshni/research/asr_multi/final_eval/scoring/score.sh"
    subprocess.call("cd " + out_dir + "; sh " + score_script_loc + " " + out_file, shell=True)


if __name__=="__main__":
    args = parse_options()
    grid_search(args)