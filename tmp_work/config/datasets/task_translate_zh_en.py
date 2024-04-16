from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassArenaDataset

subjective_reader_cfg = dict(
    input_columns=['question', 'ref'],
    output_column='judge',
    )

data_path = "/mnt/petrelfs/linjunyao/projects/opencompass/tmp_work/data/"

subjective_datasets = []


llm_tasks_w_def_zh_prompt = """
You are an specialized translator in the field of large language model research. Please translate the following LLM task category and definition from chinese to english. Use the present tense instead of present continuous tense. Paraphrase if necessary.

{question}
"""


llm_tasks_prompt = """
你是一个大模型训练专家，需要辅助人类专家对大模型预训练数据的题目进行任务分类。请根据以下任务类别写一段大模型在该类别需要完成的任务描述。长度控制在30字左右。

[任务类别]
{question}
"""


sub_map = {
    "llm_tasks_w_def_zh": llm_tasks_w_def_zh_prompt,
    # "llm_tasks": llm_tasks_prompt,
}

for _name, _prompt in sub_map.items():
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt=_prompt
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_seq_len=4096, 
            max_out_len=2048, # set as 1024 for yi models
            ),
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
        ))
