# coding=utf8

import os
import json
import log_parser

def main(dir_path, out_path):
    res = dict()
    f_names = os.listdir(dir_path)
    for f_name in f_names:
        if f_name.endswith(".sqlite"):
            f_path = os.path.join(dir_path, f_name)
            tot_time, util, nccl_ratio, mem_ratio = log_parser.do_parse(f_path)
            if tot_time is not None:
                key = f_name.rstrip(".sqlite")
                res[key] = dict()
                res[key]['tot'] = tot_time / 4e+6
                res[key]['util'] = util
                res[key]['nccl'] = nccl_ratio
                res[key]['mem'] = mem_ratio

    with open(out_path, 'w') as out_:
        json.dump(res, out_)



if __name__ == '__main__':
    import sys
    input_dir = sys.argv[1]
    out_path = sys.argv[2]
    main(input_dir, out_path)



