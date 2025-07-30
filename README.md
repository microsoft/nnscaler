*SOSP'25 Artifact Evaluation for VERDICT*

# Introduction to Verdict
**Overview:**
**Verdict** is a tool to verify parallelization equivalence for distributed model training. It ensures that the original and parallelized models are arithmetically equivalent for each training iteration, effectively eliminating bugs such as wrong tensor transformation, wrong collective communications, etc., that are introduced in the parallelization process.

**Workflow:**
Verdict takes the *execution plans* of both single-device (original) model and multi-device (parallelized) model as inputs enriched with *lineages*, and converts respective execplan into *symbolic SSA DAG* (single-static assignment directed acyclic graph). Verdict then partition dual graphs into small subgraphs to form independent *stages* based on lineages. Once stages are determined, they are executed in parallel. Within each stage, tensor shape reduction is applied, and z3 is used to symbolically verifies the output equivalence. Once all stage passes the check, the end-to-end equivalence is guaranteed.
![DesignOverview](docs/assets/design.png)

**Implementation:** 
Verdict is implemented with an interface design. Verdict defines a general graph interface and a solver interface, with a list of allowable reigistered operators. The nnScaler backend takes responsibility to produce SSA DAGs that meet the requirements. A coarse reference to source code is shown in below.
![ImplOverview](docs/assets/impl.png)

# Artifact Evaluation Guide
Welcome to artifact evaluation guide for Verdict(SOSP'25). The following document outlines the procedures needed to reproduce our results and guides you through the key experiments presented in the paper.


### ✅ Checklist with Estimated Time Cost
1. Access hardware resources. (Azure, clone repo)
2. Installation. (conda environment, demo runs)
⏱️ 10 mins
3. Run *Real-world Parallelization* Evaluation (§8.1)
⏱️ 24 hours
4. Run *Scalability* Evaluation (§8.2)
⏱️ 6 hours
5. Run *Bug Reproduction* Evaluation (§8.3)
⏱️ 20 mins

> ⚠️ **Note:** Due to refined design, optimizations and code refactoring, the current evaluation results are improved thus different from statistics in the submitted paper. Please refer to each section for expected outputs. 

### 💻 Hardware Requirements
To fully reproduce results, we recommend to run Verdict artifact evaluation on machines with at least 32 CPU (virtual) cores and 1TB memory. For SOSP'25 AE reviewers, please contact authors for virtual machine instances.

## 🚀 Installation
1. Create conda environment.
    ```
    cd Verdict
    conda env create -f conda-environment.yml
    conda activate verdict
    ```
2. Run demo verification for 2-layer llama3 model parallelization.
    ```
    bash scripts/demo_llama3.sh 
    ```
    The `scripts/demo_llama3.sh` essentially runs the following command:
    ```
    python main.py \
        --sm gen_model/mgeners/llama3_default_dp1_pp1_tp1_nm1_gbs32_ly32_h32_hi128_sq8192.pkl \
        --pm gen_model/mgeners/llama3_default_dp2_pp2_tp2_nm2_gbs32_ly32_h32_hi128_sq8192.pkl \
        --seed 0 \
        --time  \
        --max_ser_proc 30 \
        --max_vrf_proc 30 \
        --loglevel INFO \
        --no_cache_nodes \
        --no_cache_stages \
        |& tee -a data/logs/llama3_default_dp2_pp2_tp2_nm2_gbs32_ly32_h32_hi128_sq8192.txt
    ```
    > Command interpretation: `main.py` is the entry of Verdict. `--sm` and `--pm` sepcify the paths of single-device model's and parallelized model's execution plan respectively. `--seed` sets z3 random seed. `--time` activates timer. `--max_ser_proc` and `--max_vrf_proc` set the multiprocessing pool size for building SSA DAGs, and parallel stage execution respectively. `--loglevel` sets logger level. `--no_cache_nodes` and `--no_cache_stages` ignore any cached data and run verification from scratch. `|& tee -a ...` writes logs to a file for inspection convenience.
    
    **👀 Expected Output:** The program should print the following message or similar at the bottom of the output. Indicating successful execution of all stages, as well as the verified end-to-end equivalence.
    ```
    parallel verifying stages: 100%|██████████| 3909/3909 [00:05<00:00, 676.09it/s] 
    PID: ... - ✅ SUCCESS 
    Stats(success=True, ... )
    ```
    > A failed run would print `PID: ... - ❌ FAIL`.
3. Once the demo runs successfully, we can move on to reproduce evaluations.

## 🚀 Evaluate *Real-World Parallelization*

### 🎯 Goal

### ⏳ Estimated Completion Time

### 🛠 How to Run

### 👀 Expected Output