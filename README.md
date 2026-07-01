# DeFrame

Official implementation of **DeFrame: Framing-aware Debiasing for LLM Fairness**.
Paper: [DeFrame: Framing-aware Debiasing for LLM Fairness](https://arxiv.org/abs/2602.04306)

This repository contains code for running DeFrame experiments on fairness-related benchmarks and evaluating model responses under different framing conditions.

## Repository Structure

```text
DeFrame/
├── DeFrame.py
├── analysis/
│   ├── result_BBQ.ipynb
│   ├── result_DoNotAnswer_Framed.ipynb
│   ├── result_70Decisions_Framed.ipynb
│   ├── bbq_analysis_utils.py
├── benchmark/
│   ├── DNA
│   │   ├── DNA_x5.pkl
│   ├── 70decisions_explicit
│   │   ├── 70decisions.pkl
│   ├── 70decisions_implicit
│   │   ├── 70decisions.pkl
├── results/
└── README.md
```

## Setup

First, clone this repository:

```bash
git clone https://github.com/KaheeLim/DeFrame.git
cd DeFrame
```

## Download External Benchmarks

Some benchmark datasets should be cloned separately.

### BBQ

Clone the BBQ repository in the main directory of this project:

```bash
git clone https://github.com/nyu-mll/BBQ.git
```


### Do-Not-Answer

Clone the Do-Not-Answer repository inside the `analysis/` directory:

```bash
cd analysis
git clone https://github.com/Libr-AI/do-not-answer.git
```

After cloning, the directory structure should look like this:

```text
DeFrame/
├── analysis/
│   ├── do-not-answer/
│   └── ...
├── DeFrame.py
└── ...
```

## Running DeFrame

You can run experiments using `DeFrame.py`.


### Run DeFrame - BBQ

```bash
python DeFrame.py --BENCHMARK BBQ --MODEL Qwen/Qwen2.5-3B-Instruct
```

### Run DeFrame - DoNotAnswer_Framed

```bash
python DeFrame.py --BENCHMARK donotanswer_framed --MODEL Qwen/Qwen2.5-3B-Instruct
```

### Run DeFrame - 70Decisions_explicit_Framed

```bash
python DeFrame.py --BENCHMARK 70Decisions_explicit_framed --MODEL Qwen/Qwen2.5-3B-Instruct
```

### Run analysis

You can analyze your experimental results using the `.ipynb` files in the `analysis/` directory.

## Notes

* The BBQ repository should be cloned in the main project directory.
* The Do-Not-Answer repository should be cloned inside the `analysis/` directory.
* The experiments in the paper were conducted using Ollama, while this repository implements inference with vLLM; therefore, the results may not exactly match those reported in the paper.
