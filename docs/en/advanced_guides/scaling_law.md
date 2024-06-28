# Scaling Law Evaluation

## Introduction

The following introduction comes from the abstract of [Compression Represents Intelligence Linearly](https://arxiv.org/abs/2404.09937):

>There is a belief that learning to compress well will lead to intelligence. Recently, language modeling has been shown to be equivalent to compression, which offers a compelling rationale for the success of large language models (LLMs): the development of more advanced language models is essentially enhancing compression which facilitates intelligence. ...our findings suggest that compression efficiency, as an unsupervised metric derived from raw text corpora, serves as a reliable evaluation measure that is linearly associated with the model capabilities. We open-source our compression datasets as well as our data collection pipelines to facilitate future researchers to assess compression properly.

## Step-1: Data preparation

The first step is to download and prepare your dataset.

### Dataset
The dataset, which consists of three external corpora, can be downloaded using the following python script:

```python
from os import os.path as osp
from datasets import load_dataset

data_path = "data/llm-compression"

subset_mapping = {
    'arxiv_math': ['arxiv_math'],
    'commoncraw': ['cc'],
    'python': ['python'],
}

for key, value in subset_mapping.items():
    llmc_dataset = load_dataset(r"hkust-nlp/llm-compression", name=value)
    llmc_dataset["test"].to_json(osp.join(data_path, f"{key}.jsonl"))
```

Note: Refer to the original [repository](https://github.com/hkust-nlp/llm-compression-intelligence) for more details on data collection and design.

### Dataset configuration

The dataset configuration file (`configs/datasets/llm-compression.py`) sets the inferencer, evaluator, and data loader for evaluation:

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import SWCELossInferencer
from opencompass.openicl.icl_evaluator import BPCEvaluator
from opencompass.datasets import LLMCompressionDataset


# The three corpora for llm_compression used in the original paper
# See configs/datasets/llm_compression/README.md for more details
subset_mapping = {
    'arxiv_math': ['arxiv_math'],
    'commoncraw': ['cc'],
    'python': ['python'],
}


# Build LLM Compression datasets
llm_compression_datasets = []
for _name in subset_mapping.keys():
    llm_cmp_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template='{content}',
        ),
        # No in-context example, using ZeroRetriever
        retriever=dict(type=ZeroRetriever),
        # Calculates cross entropy loss for each batch based on a sliding context window
        # Setting block_size=1900 and stride=512 according to the original paper
        inferencer=dict(type=SWCELossInferencer, block_size=1900, stride=512),
    )

    # Calculates Bits per Character (BPC) based on the CE loss from the inference stage
    llm_cmp_eval_cfg = dict(evaluator=dict(type=BPCEvaluator))

    llm_compression_datasets.append(
        dict(
            abbr=f'llm_compression-{_name}',
            type=LLMCompressionDataset,
            path='./data/llm-compression',
            name=_name,
            samples=None,  # Set small samples for testing
            reader_cfg=dict(
                input_columns=['content'],
                output_column=None,
            ),
            infer_cfg=llm_cmp_infer_cfg,
            eval_cfg=llm_cmp_eval_cfg,
        ))
```

Here we use the `ZeroRetriever` since there are no in-context examples for each 'prompt'. We use `SWCELossInferencer` for the inference stage to calculate the cross entropy loss for each batch of tokens based on a sliding context window, setting `block_size=1900` (window length) and `stride=512` (step size).