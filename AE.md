# Artifact Evaluation

# Install

* Install the repo on all servers:

```sh
conda create -n cupilot python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
cd cupilot && pip install -e . && cd ..
cd Tessel && pip install -e . && cd ..
pip install deepspeed
```

# Run Experiments

For experiments requiring more than 1 servers, e.g., `xx_32gpus.sh`, please setup `$MASTER_ADDR` and `$NODE_RANK` for each server.

* Swin-Transformer:

```sh
cd cupilot
bash evaluation/swin_4gpus.sh
bash evaluation/swin_8gpus.sh
bash evaluation/swin_16gpus.sh
bash evaluation/swin_32gpus.sh
```

* T5:

```sh
cd cupilot
bash evaluation/t5_4gpus.sh
bash evaluation/t5_8gpus.sh
bash evaluation/t5_16gpus.sh
bash evaluation/t5_32gpus.sh
```

Please find all the evaluation results in the `logs/` folder.

