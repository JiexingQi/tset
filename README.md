# TSET

Implementations for paper "Enhancing SPARQL Query Generation for KBQA Systems by Learning to Correct Triplets".

## Environment setup

Create an environment using  **Python 3.7**, and install the dependencies with

```
pip install -r requirements.txt
```

## Dataset preparation

### Use pre-processed datasets

The pre-processed datasets that can be used directly for training has been placed under the folder *transform/transformers_cache/downloads*. And we recommend you to use them.

- **LC-QuAD2.0-master:** Processed LC-QuAD 2.0 dataset for fine-tuning;
- **LC-QuAD2.0-pre:** Processed LC-QuAD 2.0 dataset for pre-training;
- **QALD_9_PULS:** Processed Qald-9-plus dataset for fine-tuning;
- **QALD_10:** Processed Qald-10 dataset for fine-tuning.

### Do yourself

You can also download the original datasets and process them yourself. 

- LC-QuAD 2.0, [link](https://github.com/debayan/gett-qa/tree/main/lcquad2/dataset)
- Qald-9-plus, [link](https://github.com/KGQA/qald_9_plus)
- Qald-10, [link](https://github.com/KGQA/QALD-10/tree/main)

The preprocessing scripts are under the folder *preprocess/LC-QuAD2.0-pre*.

## Pre-training

The *configs/train_1.json* is an example of parameter configuration for pre-training.

Replace `"model_name_or_path"` with the model name (`t5-small`, `t5-base`, or `t5-large`) or  the path to your checkpoint ,  `"output_dir"` with where you want to store your outputs,  and `"cache_dir"` with the place for caching. 

You can simply run the code below:

```
CUDA_VISIBLE_DEVICES=0 python seq2seq/run_seq2seq.py configs/train_1.json
```

## Fine-tuning

The *configs/train_2.json* is an example of parameter configuration for fine-tuning the model.

You should replace `"dataset"` with the name of the dataset that your want to fine-tune the model on, and you can choose from `[lc_quad_2, qald_9, qald_10]`. Replace `"model_name_or_path"` with the path to your checkpoint obtained during the previous pre-training.

You can simply run the code below:

```
CUDA_VISIBLE_DEVICES=0 python seq2seq/run_seq2seq.py configs/train_2.json
```

