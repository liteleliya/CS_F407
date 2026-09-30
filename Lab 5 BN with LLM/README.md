# Lab 5: Bayesian Networks with an LLM

The course notebook `notebooks/llm_bn.ipynb`, run end to end with a local coding LLM (`Qwen/Qwen2.5-Coder-1.5B-Instruct`), with every exercise completed. Everything I added is marked **[Added]** in the notebook.

## Files

- `llm_bn_completed.ipynb` - the executed notebook with all outputs, inspection notes, revisions, the HW exercise, the classroom exercises, and a summary.

## What is in it

- The trusted Sprinkler network, exact inference, and an enumeration check.
- LLM-generated code for inference, MLE, and Bayesian (BDeu) estimation: inspected, executed, and validated against trusted results. The first attempts failed (a removed pgmpy API, wrong CPT shapes and values, a hidden file read). The notebook documents how the prompt was revised and the one manual fix needed.
- An extended static safety check that catches file reads done through `pd.read_csv`.
- HW exercise: P(S=1 | W=1), P(C=1 | W=1), P(R=1 | W=1, S=0), with predictions made before running, and checks against variable elimination and full enumeration.
- Classroom exercises: the LLM's explanation of the WetGrass CPT columns, checked against pgmpy (the LLM got it wrong), and sampling variability across datasets, checked against the predicted standard error.

## Running it

Needs `pgmpy>=1.0`, `transformers`, `torch`, `pandas`, `matplotlib`, and about 3 GB for the Qwen model on first run.

```bash
jupyter nbconvert --to notebook --execute llm_bn_completed.ipynb --output llm_bn_completed.ipynb
```
