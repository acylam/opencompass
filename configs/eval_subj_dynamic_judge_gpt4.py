import os
# from os import getenv as gv
# from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base

with read_base():
    # from .datasets.subjective.compassarena.compassarena_compare import subjective_datasets
    from .datasets.subjective.chain_of_thought.cot import subjective_datasets

    # internlm
    from .models.hf_internlm.hf_internlm2_chat_20b import models as internlm2_20b_chat
    from .models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_7b_chat

    # mistral 7b
    from .models.mistral.hf_mistral_7b_instruct_v0_2 import models as mistral_7b

    # mixtral 8x7b
    from .models.mixtral.hf_mixtral_8x7b_instruct_v0_1 import models as mixtral_8x7b

    # qianwen
    from .models.qwen.hf_qwen1_5_7b_chat import models as qwen2_7b_chat
    from .models.qwen.hf_qwen1_5_14b_chat import models as qwen2_14b_chat
    from .models.qwen.hf_qwen1_5_72b_chat import models as qwen2_72b_chat
    from .models.deepseek.hf_deepseek_67b_chat import models as deepseek_67b_chat_hf

    # yi
    from .models.yi.hf_yi_34b_chat import models as yi_34b_chat

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI, ZhiPuV2AI, ERNIEBot, Qwen
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

# -------------Inference Stage ----------------------------------------

azure_gpt4 = dict(
    abbr='azure-gpt4-turbo',
    # type=OpenAI,
    type=OpenAIAllesAPIN,
    url='http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/openai/v2/text/chat',
    path='gpt-4-1106-preview',
    key=os.getenv('ALLES_APIN_OPENAI_KEY'),
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=1024,
    # max_seq_len=4096,
    max_seq_len=2048,
    batch_size=2,
    retry=20,
    temperature=1, # more creative for inference
)

glm4_notools = dict(
    abbr="glm4_notools",
    type=ZhiPuV2AI,
    path="glm-4",
    key=os.getenv('ZHIPU_API_KEY'),  # opencompass
    generation_kwargs={
        "tools": [{
                "type": "web_search", 
                "web_search": {"enable": False}
        }]
    },
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=2048,
    # max_seq_len=16,
    batch_size=4,
)

qwen_max = dict(
    abbr='qwen-max',
    retry=10,
    type=Qwen,
    path='qwen-max',
    key=os.getenv('QWEN_API_KEY'), # please give your key
    generation_kwargs={
        'enable_search': False,
    },
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=2048,
    batch_size=8
)

erniebot_pro = dict(
    abbr='erniebot_pro', # 4.0
    type=ERNIEBot,
    path='erniebot_pro',
    retry=5,
    key=os.getenv('ERNIEBOT_API_KEY'),
    secretkey=os.getenv('ERNIEBOT_SECRET_KEY'),
    url='https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=',
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=2048,
    max_seq_len=2048,
    batch_size=4
)

datasets = [
    *subjective_datasets
]

hf_models = [
    *internlm2_7b_chat,
    *internlm2_20b_chat, 
    *qwen2_7b_chat,
    *qwen2_14b_chat,
    *qwen2_72b_chat,
    *deepseek_67b_chat_hf,
    *yi_34b_chat,
    *mistral_7b,
    *mixtral_8x7b,
]

# Add additional models from predictions
api_models = [
    # azure_gpt4,
    glm4_notools,
    qwen_max,
    erniebot_pro,
]

models = []
for mdl in hf_models:
    # if 'generation_kwargs' not in mdl:
    #     mdl['generation_kwargs'] = {'do_sample': True}
    # else:
    #     mdl['generation_kwargs']['do_sample'] = True
    models.append(mdl)

models.extend(api_models)

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
        partition='llm_dev2',
        quotatype='reserved',
        type=SlurmSequentialRunner,
        task=dict(type=OpenICLInferTask),  # Task to be run
        max_num_workers=256,  # Maximum concurrent evaluation task count
        retry=2,  # Retry count for failed tasks, can prevent accidental errors
    ),
)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_model = dict(
    abbr='azure-gpt4-turbo',
    # type=OpenAI,
    type=OpenAIAllesAPIN,
    url='http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/openai/v2/text/chat',
    path='gpt-4-1106-preview',
    key=os.getenv('ALLES_APIN_OPENAI_KEY'),
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=1024,
    # max_seq_len=4096,
    max_seq_len=2048,
    batch_size=2,
    retry=20,
    temperature=0,
)

# judge_model = dict(
#     abbr="glm4",
#     type=ZhiPuV2AI,
#     path="glm-4",
#     key=os.getenv('ZHIPU_API_KEY'),
#     generation_kwargs={"tools": [{"type": "web_search", "web_search": {"enable": False}}]},
#     meta_template=api_meta_template,
#     query_per_second=1,
#     max_out_len=2048,
#     max_seq_len=2048,
#     batch_size=8,
# )

# ## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        strategy='split',
        max_task_size=10000,
        # mode='singlescore',
        mode='m2n',
        # models=models,
        # base_models=[gpt4],
        base_models=[azure_gpt4],
        compare_models=models,
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='reserved',
        max_num_workers=32,
        task=dict(type=SubjectiveEvalTask, judge_cfg=judge_model),
    ),
)

workdir = "outputs/subj_dynamic_judge_gpt4/"

summarizer = dict(type=CompassArenaSummarizer, summary_type='half_add')
