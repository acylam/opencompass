# from os import getenv as gv
# from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base

with read_base():
    # from .datasets.siqa.siqa_gen import siqa_datasets
    # from .datasets.winograd.winograd_ppl import winograd_datasets
    from .datasets.ceval.ceval_gen import ceval_datasets
    # from .datasets.subjective.compassarena.compassarena_compare import subjective_datasets
    # from .models.opt.hf_opt_125m import opt125m
    # from .models.opt.hf_opt_350m import opt350m

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI, ZhiPuV2AI
from opencompass.models.openai_api import OpenAIAllesAPIN
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import CompassArenaSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

# -------------Inference Stage ----------------------------------------

glm4 = dict(
    abbr="glm4",
    type=ZhiPuV2AI,
    path="glm-4",
    key="0a551f63dba0048c095c2f35c1a04aef.uvGOi5fmBDDpjhZP",  # opencompass
    # key='58d38c0f9784fd9cf9cc520f6d41e3b2.NJzXtDpuYcElAAeN',
    # key=
    # 'a66000a0776c862931176b3ef2e26faf.HbE8LB5Lz152SeVr',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    generation_kwargs={"tools": [{"type": "web_search", "web_search": {"enable": False}}]},
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=2048,
    # max_seq_len=16,
    batch_size=8,
)

# datasets = [*siqa_datasets, *winograd_datasets]
# models = [opt125m, opt350m]

# datasets = [*ceval_datasets]
datasets = [*subjective_datasets]
models = [glm4]
# models = []

# infer = dict(
#     partitioner=dict(type=SizePartitioner, strategy='split', max_task_size=10000),
#     runner=dict(
#         type=SlurmSequentialRunner,
#         partition='llm_dev2',
#         quotatype='auto',
#         max_num_workers=256,
#         task=dict(type=OpenICLInferTask),
#     ),
# )

# infer = dict(
#     partitioner=dict(type=SizePartitioner, strategy='split', max_task_size=10000),
#     # partitioner=dict(type=NaivePartitioner)
#     runner=dict(
#         type=LocalRunner,
#         # partition='llmeval',
#         # quotatype='auto',
#         max_num_workers=256,
#         task=dict(type=OpenICLInferTask),
#     ),
# )

infer = dict(
    partitioner=dict(type=SizePartitioner, strategy='split', max_task_size=10000),
    runner=dict(
        partition='llmeval',
        quotatype='auto',
        type=SlurmSequentialRunner,
        task=dict(type=OpenICLInferTask),  # Task to be run
        max_num_workers=256,  # Maximum concurrent evaluation task count
        retry=2,  # Retry count for failed tasks, can prevent accidental errors
    ),
)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
# judge_model = dict(
#     abbr='GPT4-Turbo',
#     type=OpenAI,
#     path='gpt-4-1106-preview',
#     key='',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
#     meta_template=api_meta_template,
#     query_per_second=1,
#     max_out_len=1024,
#     max_seq_len=4096,
#     batch_size=2,
#     retry=20,
#     temperature=0,
# )

judge_model = dict(
    abbr="glm4",
    type=ZhiPuV2AI,
    path="glm-4",
    key="0a551f63dba0048c095c2f35c1a04aef.uvGOi5fmBDDpjhZP",  # opencompass
    # key='58d38c0f9784fd9cf9cc520f6d41e3b2.NJzXtDpuYcElAAeN',
    # key=
    # 'a66000a0776c862931176b3ef2e26faf.HbE8LB5Lz152SeVr',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
    generation_kwargs={"tools": [{"type": "web_search", "web_search": {"enable": False}}]},
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    # max_seq_len=2048,
    max_seq_len=2048,
    batch_size=8,
)

# ## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        strategy='split',
        max_task_size=10000,
        mode='m2n',
        base_models=[glm4],
        compare_models=models,
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=32,
        task=dict(type=SubjectiveEvalTask, judge_cfg=judge_model),
    ),
)

# workdir = "outputs/glm4_ceval/"
# workdir = "outputs/glm4_subjective_eval"

summarizer = dict(type=CompassArenaSummarizer, summary_type='half_add')
