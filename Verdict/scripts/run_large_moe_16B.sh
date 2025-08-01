#!/bin/bash
python main.py --sm gen_model/mgeners/moe_default_dp1_pp1_tp1_nm1_gbs1024_ly27_h16_hi512_sq128_a6_r64.pkl --pm gen_model/mgeners/moe_default_dp16_pp2_tp4_nm16_gbs1024_ly27_h16_hi512_sq128_a6_r64.pkl --seed 0 --no_cache_nodes --no_cache_stages --time  --max_ser_proc 30 --max_vrf_proc 30 --loglevel INFO |& tee -a data/logs/moe_default_dp16_pp2_tp4_nm16_gbs1024_ly27_h16_hi512_sq128_a6_r64.txt

