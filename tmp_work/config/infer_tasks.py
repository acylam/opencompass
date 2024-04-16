import os
# from os import getenv as gv
# from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base

with read_base():
    # from .datasets.subjective.compassarena.compassarena_compare import subjective_datasets
    # from .datasets.task_categories import subjective_datasets
    from .datasets.question_difficulty import subjective_datasets

    # internlm
    # from .models.hf_internlm.hf_internlm2_chat_20b import models as internlm2_20b_chat
    # from .models.hf_internlm.hf_internlm2_chat_7b import models as internlm2_7b_chat

    # # qianwen
    # from .models.qwen.hf_qwen1_5_7b_chat import models as qwen2_7b_chat
    # from .models.qwen.hf_qwen1_5_14b_chat import models as qwen2_14b_chat
    # from .models.qwen.hf_qwen1_5_72b_chat import models as qwen2_72b_chat
    # from .models.deepseek.hf_deepseek_67b_chat import models as deepseek_67b_chat_hf

    # # yi
    # from .models.yi.hf_yi_34b_chat import models as yi_34b_chat


from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI, ZhiPuV2AI, Qwen, ERNIEBot
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

datasets = [
    *subjective_datasets
]

azure_gpt4 = dict(
    abbr='azure-gpt4-turbo',
    # type=OpenAI,
    type=OpenAIAllesAPIN,
    url='http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/openai/v2/text/chat',
    path='gpt-4-1106-preview',
    os.getenv('ALLES_APIN_OPENAI_KEY'),
    meta_template=api_meta_template,
    query_per_second=1,
    max_out_len=1024,
    max_seq_len=4096,
    # max_seq_len=2048,
    batch_size=8,
    retry=20,
    temperature=0.85, # more creative for inference
)

glm4_notools = dict(
    abbr="glm4_notools",
    type=ZhiPuV2AI,
    path="glm-4",
    key=os.getenv('ZHIPU_API_KEY'),
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
    batch_size=8,
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

hf_models = [
    # *internlm2_7b_chat,
    # *internlm2_20b_chat, 
    # *qwen2_7b_chat,
    # *qwen2_14b_chat,
    # *qwen2_72b_chat,
    # *deepseek_67b_chat_hf,
    # *yi_34b_chat,
]

# Add additional models from predictions
api_models = [
    azure_gpt4,
    # glm4_notools,
    # qwen_max,
    # erniebot_pro,
]

models = []
for mdl in hf_models:
    mdl['generation_kwargs']['do_sample'] = True
    models.append(mdl)

models.extend(api_models)

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

workdir = "/mnt/petrelfs/linjunyao/projects/opencompass/tmp_work/outputs/infer_tasks"
