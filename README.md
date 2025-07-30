# SOSP'25 Artifact Evaluation for VERDICT

## Introduction to Verdict
**Overview:**
**Verdict** is a tool to verify parallelization equivalence for distributed model training. It ensures that the original and parallelized models are arithmetically equivalent for each training iteration, effectively eliminating bugs such as wrong tensor transformation, wrong collective communications, etc., that are introduced in the parallelization process.

**Workflow:**
Verdict takes the *execution plans* of both single-device (original) model and multi-device (parallelized) model as inputs enriched with *lineages*, and converts respective execplan into *symbolic SSA DAG* (single-static assignment directed acyclic graph). Verdict then partition dual graphs into small subgraphs to form independent *stages* based on lineages. Once stages are determined, they are executed in parallel. Within each stage, tensor shape reduction is applied, and z3 is used to symbolically verifies the output equivalence. Once all stage passes the check, the end-to-end equivalence is guaranteed.
![DesignOverview](docs/assets/design.png)

**Implementation:** 
Verdict is implemented with an interface design. Verdict defines a general graph interface and a solver interface, with a list of allowable reigistered operators. The nnScaler backend takes responsibility to produce SSA DAGs that meet the requirements. A rough guidance to source code is shown in below.
![ImplOverview](docs/assets/impl.png)

## Artifact Evaluation Guide
Welcome to artifact evaluation guide for Verdict(SOSP'25). The following document outlines the procedures needed to reproduce our results and guides you through the key experiments presented in the paper.

### ✅ Checklist with Estimated Time Cost
- [] Access hardware resources. (Azure, clone repo)
- [] Environment setup. (conda environment, demo runs)
⏱️ 10 mins
- [] Run Large Training Evaluation (§8.1)
⏱️ 24 hours
- [] Run Scalability Evaluation (§8.2)
⏱️ 6 hours
- [] Run Bug Reproduction Evaluation (§8.3)
⏱️ 20 mins

### 💻 Hardware Requirements