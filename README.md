````markdown
# LLM Peer Review Bias Evaluation

This repository contains the experimental code, generated variants, and result files for evaluating bias in large language models (LLMs) when they are used as simulated peer reviewers.

The project studies whether LLM review scores change when the same paper is presented with different author identity or affiliation signals. The core idea is to keep the paper content fixed while modifying controlled metadata such as author ethnicity, institutional prestige, or income-level affiliation context.

---

## Repository Structure

```text
llm-peer-review/
├── experiments/
│   ├── ethnicity-bias/
│   ├── prestige-bias/
│   └── income-bias/
│
├── data/
│   ├── processed_papers/
│   └── metadata/
│
├── README.md
└── LICENSE
````

---

## Experiments

### 1. Ethnicity Bias

This experiment evaluates whether LLMs exhibit bias in peer-review scoring based on author ethnicity.

Location:

```text
experiments/ethnicity-bias/
```

Results:

```text
exp1_ethnicity_binary.xlsx
exp1_ethnicity_6point.xlsx
```

---

### 2. Prestige Bias

This experiment evaluates whether LLMs exhibit bias based on author prestige by comparing well-known researchers with assigned author identities.

Location:

```text
experiments/prestige-bias/
```

Results:

```text
exp2_prestige_binary.xlsx
exp2_prestige_6point.xlsx
```

---

### 3. Income Bias

This experiment evaluates whether LLMs exhibit bias based on the economic context of author affiliations, such as high-income versus low-income institutional settings.

Location:

```text
experiments/income-bias/
```

Results:

```text
exp3_income_binary.xlsx
exp3_income_6point.xlsx
```

---

## Common Experiment Structure

Each experiment follows the same structure:

```text
experiment-name/
├── README.md
├── scripts/
│   ├── build_variants.py
│   └── run_evaluations.py
├── variants/
│   ├── index.csv
│   └── <group>/<author_id>/<paper>.md
└── results/
    ├── *_binary.xlsx
    └── *_6point.xlsx
```

* `scripts/` contains code for variant generation and LLM evaluation.
* `variants/` contains generated paper variants and the corresponding `index.csv`.
* `results/` contains evaluation outputs for binary and 6-point scoring settings.
* Each experiment has its own local `README.md`.

---

## Evaluation Settings

Two review scoring protocols are used:

### Binary Evaluation

```text
1 = Definite Reject
2 = Accept as Is
```

### 6-Point Evaluation

```text
1 = Definite Reject
2 = Reject with Option to Resubmit
3 = Major Revision
4 = Minor Revision
5 = Very Minor Revision
6 = Accept as Is
```

Each variant is evaluated multiple times to reduce output variability.

---

## Models Evaluated

The experiments evaluate multiple LLMs through OpenRouter, including:

```text
Gemma 3 27B
LLaMA 4 Scout
DeepSeek V3
```

---

## Data

The `data/` folder contains metadata and processed paper files used to generate the variants.

```text
data/
├── processed_papers/
└── metadata/
```

Depending on file size and copyright constraints, full raw paper files may not be included. The repository focuses on reproducible scripts, metadata structure, generated variants, and result outputs.

---

## Running the Code

Before running evaluation scripts, set the OpenRouter API key:

```bash
setx OPENROUTER_API_KEY "your_key_here"
```

Then run the relevant scripts from the experiment folder:

```bash
python scripts/build_variants.py
python scripts/run_evaluations.py
```

Each experiment is self-contained and can be reproduced independently.

---

## Notes

* The paper content is kept fixed across variants.
* Only controlled author or affiliation metadata is changed.
* Results include per-run scores, average scores, and token usage where available.
* Binary and 6-point settings are stored separately for each experiment.

---

## License

This repository is released under the license provided in the `LICENSE` file.

```
```
