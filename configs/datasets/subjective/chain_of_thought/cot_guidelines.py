from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassArenaDataset

subjective_reader_cfg = dict(
    input_columns=['question', 'ref'],
    output_column='judge',
    )

data_path = "data/subjective/compass_arena"
# data_path = "/mnt/petrelfs/share_data/caomaosong/subjective/compass_arena"

subjective_datasets = []

creation_zh_prompt = """
现在你要扮演一个人类专家，对大模型模型在具体问题上的回复进行评价，请针对<QuestionStart><QustionEnd>之间的问题设计5条具体、清晰、明确且具有相关性的评估标准。请你在<GuidelineStart><GuidelineEnd>之间逐条列出你的评估标准。

<QuestionStart>
{{question}}
<QustionEnd>
"""

creation_en_prompt = """
You are an expert in large language model evaluation. Based on the question provided within <QuestionStart><QustionEnd>, design 5 specific, clear, and relevant evaluation metrics to help guide the task of evaluating answer quality. Your 5 metrics should be listed within <GuidelineStart><GuidelineEnd>.

<QuestionStart>
{{question}}
<QustionEnd>
"""

sub_map = {
    "creationv2_zh": creation_zh_prompt,
    # "creationv2_en": creation_en_prompt,
}

for _name, _prompt in sub_map.items():
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=creation_zh_prompt,
                ),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_seq_len=4096, max_out_len=2048),
    )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            infer_order='double',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = _prompt
                    ),
                ]),
            ),
        ),
        pred_role="BOT",
    )

    subjective_datasets.append(
        dict(
            abbr=f"{_name}",
            type=CompassArenaDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        )
    )
