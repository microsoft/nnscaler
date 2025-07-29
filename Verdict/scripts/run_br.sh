# echo correct baseline llama3 8B
# python main.py --dump_nodes --dump_lineages --dump_stages --no_cache_nodes --no_cache_stages --loglevel INFO --sm gen_model/mgeners/llama3_default_dp1_pp1_tp1_nm1_gbs32_ly32_h32_hi128_sq8192.pkl --pm gen_model/mgeners/llama3_default_dp2_pp2_tp2_nm2_gbs32_ly32_h32_hi128_sq8192.pkl


for num in {1..14}; do
    echo "🚀🚀🚀 BR$num"
    python main.py --dump_nodes --dump_lineages --dump_stages --no_cache_nodes --no_cache_stages --loglevel INFO  --sm gen_model/mgeners/llama3_default_dp1_pp1_tp1_nm1_gbs32_ly32_h32_hi128_sq8192.pkl --pm gen_model/mgeners/llama3_br${num}_dp2_pp2_tp2_nm2_gbs32_ly32_h32_hi128_sq8192.pkl
done