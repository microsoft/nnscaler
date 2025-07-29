#!/bin/bash
python main.py --sm gen_model/mgeners/llama3_default_dp1_pp1_tp1_nm1_gbs512_ly80_h64_hi4096_sq128.pkl --pm gen_model/mgeners/llama3_default_dp16_pp4_tp8_nm32_gbs512_ly80_h64_hi4096_sq128.pkl --seed 0 --no_cache_nodes --no_cache_stages --time  --max_ser_proc 30 --max_vrf_proc 30 --loglevel INFO |& tee -a data/logs/llama3_default_dp16_pp4_tp8_nm32_gbs512_ly80_h64_hi4096_sq128.txt

# use cache and DEBUG logger
# python main.py --sm gen_model/mgeners/llama3_default_dp1_pp1_tp1_nm1_gbs512_ly80_h64_hi4096_sq128.pkl --pm gen_model/mgeners/llama3_default_dp16_pp4_tp8_nm32_gbs512_ly80_h64_hi4096_sq128.pkl --seed 0 --time  --max_ser_proc 30 --max_vrf_proc 30 --loglevel DEBUG |& tee -a data/logs/llama3_default_dp16_pp4_tp8_nm32_gbs512_ly80_h64_hi4096_sq128.txt

# reduced parallel
# python main.py --sm gen_model/mgeners/llama3_default_dp1_pp1_tp1_nm1_gbs512_ly80_h64_hi4096_sq128.pkl --pm gen_model/mgeners/llama3_default_dp16_pp4_tp8_nm32_gbs512_ly80_h64_hi4096_sq128.pkl --seed 0 --time  --max_ser_proc 15 --max_vrf_proc 15 --loglevel INFO |& tee -a data/logs/llama3_default_dp16_pp4_tp8_nm32_gbs512_ly80_h64_hi4096_sq128.txt