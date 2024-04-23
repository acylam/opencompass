# from textwrap import dedent

from opencompass.datasets import TaskCategoriesDataset, TaskDifficultiesDataset
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

subjective_reader_cfg = dict(
    # input_columns=['category', 'definition', 'ref_or_not', 'prompt'],
    input_columns=['prompt'],
    output_column='judge',
)

data_path = '/mnt/petrelfs/linjunyao/projects/opencompass/tmp_work/data/'

subjective_datasets = []

# num_levels = 4
# num_criteria = 5
# num_instructions = 1
# instruction_length = 50
# ref_length = 150
# # category = '创意性写作'
# # definition = '生成具有创造性和独特性的文本内容'
# category = '功能性写作'
# definition = '模型生成具有特定目的和格式的文本内容，如电子邮件、工作周报、项目规划、法律文书、邀请帖等。'
# category = '自然语言处理 - 文本总结'
# definition = '提炼文本核心内容，生成简洁连贯的摘要。例如博客总结、文章总结、论文总结和笔记总结等。'
# category = '重写 - 文本错误修改'
# definition = '修正给定文本中的语法、拼写及用词错误，确保语义清晰准确。'

# # category = '{category}'
# # definition = '{definition}'

# level_template = ""
# for i in range(1, num_levels + 1):
#     level_template += dedent(f"""
#     [level{i}]
#     [level{i}_criteria_start] <Your criteria here> [level{i}_criteria_end]
#     [level{i}_example_instruction_start] <Your {i*instruction_length} word example instruction here> [level{i}_example_instruction_end]
#     [level{i}_reference_text_start] <Your {i*ref_length} word reference text here> [level{i}_reference_text_end]
#     """)

# zh_prompt = f"""
# 你是一个LLM评测专家，你需要设计大模型instruction的难度标准和制造对应难度的example instruction，根据我给出的一个“特定类别”来制造{num_levels}个instruction难度级别，每个级别设计{num_criteria}个难度标准和{num_instructions}个example instruction。你制造的难度标准和example instruction必须符合以下要求：

# 1.制造的难度标准和example instruction必须符合给出的“特定类别”
# 2.制造的难度标准要具体和可量化，给每个标准提供一个标题
# 3.制造的{num_levels}个难度标准需要彼此之间具有区分度
# 4.制造的{num_levels}个难度标准需要按从易到难进行出题，随着难度等级上升而提高复杂度
# 5.制造的example instruction一定要包含当前等级的所有难度标准项，且字数不能少于{instruction_length}字
# 6.制造的example instruction不能提供可选答案
# 7.制造的难度标准和example instruction要按照一定的格式进行回复，即在每个难度级别后面需要给出对应的{num_criteria}个难度标准、{num_instructions}个example instruction和涉及的reference text:
# ```
# {level_template}
# ```
# 8.如果你的example instruction涉及到reference text（例如文本总结、文本摘要、文本润色），你必须自己想象出一个reference text

# [特定类别]：{category}
# [类别定义]：{definition}
# 下面请你遵照给出的要求，用中文生成属于[{category}]的{num_levels}个instruction难度级别、对应的{num_criteria}个难度标准和{num_instructions}个example instruction:
# """

zh_prompt = '{prompt}'
print(zh_prompt)

# 3.制造的instruction具有创造性，能想象不同场景、不同功能的instruction
# '[level1_description_start] <Your difficulty description here> [level1_description_end]'

# zh_prompt = """
# 你是一个LLM评测专家，你需要设计大模型评测问题的任务难度级别和对应的任务难度标准，根据我给出的一个“特定类别”来制造5个难度级别，每个级别制造5个难度标准和问题示例。
# 你制造的难度级别和难度标准必须符合以下要求：
# 1.制造的难度级别和标准必须符合给出的“特定任务类别”
# 2.制造的难度级别需要给出具体和可量化的难度标准
# 3.制造的难度标准应该尽量详细，并且具有创造性，能想象不同场景、不同功能的难度标准
# 4.制造的5个难度级别需要彼此之间尽量有区分度
# 5.制造的5个难度级别需要按从易到难进行出题
# 6.制造的难度级别和难度标准要按照一定的格式进行回复，即在每个难度级别后面需要给出对应的5个难度标准和问题示例:
# ```
# [level1_description_start] <Your difficulty description here> [level1_description_end]
# [level1_criteria_start] <Your criteria here> [level1_criteria_end]
# [level1_question_start] <Your criteria here> [level1_question_end]

# [level2_description_start] <Your difficulty description here> [level2_description_end]
# [level2_criteria_start] <Your criteria here> [level2_criteria_end]
# [level2_question_start] <Your question here> [level2_question_end]

# [level3_description_start] <Your difficulty description here> [level3_description_end]
# [level3_criteria_start] <Your criteria here> [level3_criteria_end]
# [level3_question_start] <Your question here> [level3_question_end]

# [level4_description_start] <Your difficulty description here> [level4_description_end]
# [level4_criteria_start] <Your criteria here> [level4_criteria_end]
# [level4_question_start] <Your question here> [level4_question_end]

# [level5_description_start] <Your difficulty description here> [level5_description_end]
# [level5_criteria_start] <Your criteria here> [level5_criteria_end]
# [level5_question_start] <Your question here> [level5_question_end]
# ```

# [特定任务类别]：创意性写作
# [类别定义]：模型生成具有创造性和独特性的文本内容
# 下面请你遵照给出的要求，生成属于[创意性写作]的5个难度级别和对应的难度标准和问题示例:
# """

# zh_prompt = """
# 你是一个LLM评测专家，你需要设计大模型评测问题的难度级别和对应的评测标准，根据我给出的一个“特定类别”来制造5个难度级别，每个级别制造5个评测标准。
# 你制造的难度级别和评测标准必须符合以下要求：
# 1.制造的难度级别和标准必须符合给出的“特定类别”
# 2.制造的难度级别需要给出具体和可量化的评测标准
# 3.制造的评测标准应该尽量详细，并且具有创造性，能想象不同场景、不同功能的评测标准
# 4.制造的5个难度级别需要彼此之间尽量有区分度
# 5.制造的5个难度级别需要按从易到难进行出题
# 6.制造的难度级别和评测标准要按照一定的格式进行回复，即在每个难度级别后面需要给出对应的5个评测标准:
# ```
# [level1_description_start] <Your question difficulty description here> [level1_description_end]
# [level1_guidelines_start] <Your guidelines here> [level1_guidelines_end]

# [level2_description_start] <Your question difficulty description here> [level2_description_end]
# [level2_guidelines_start] <Your guidelines here> [level2_guidelines_end]

# [level3_description_start] <Your question difficulty description here> [level3_description_end]
# [level3_guidelines_start] <Your guidelines here> [level3_guidelines_end]

# [level4_description_start] <Your question difficulty description here> [level4_description_end]
# [level4_guidelines_start] <Your guidelines here> [level4_guidelines_end]

# [level5_description_start] <Your question difficulty description here> [level5_description_end]
# [level5_guidelines_start] <Your guidelines here> [level5_guidelines_end]
# ```

# [特定类别]：创意性写作
# [类别定义]：模型生成具有创造性和独特性的文本内容
# 下面请你遵照给出的要求，生成属于[创意性写作]的5个难度级别和对应的评测标准:
# """

# zh_prompt = """
# 你是一个评测question制造专家，你需要制造出人类可能会对模型提出的问题，根据我给出的一个“特定类别”来制造5个question，并且这5个question应该有难易区别。
# 你制造的question必须符合以下要求：
# 1.制造的question必须符合给出的“特定类别”
# 2.制造的question不能抄袭我的题目示例，必须要与之完全不同
# 3.制造的question应该尽量详细，并且具有创造性，能想象不同场景、不同功能的question
# 4.制造的5个question需要彼此之间尽量多样化
# 5.制造的5个question需要按从易到难的5个难度级别进行出题
# 6.制造的question必须合理，是人类口吻会问出的问题
# 7.制造的question要按照一定的格式进行回复，即在每个question后面需要给出它的难度等级和描述:
# ```
# [question1_start] <Your question here> [question1_end][Level1]
# [question2_start] <Your question here> [question2_end][Level2]
# [question3_start] <Your question here> [question3_end][Level3]
# [question4_start] <Your question here> [question4_end][Level4]
# [question5_start] <Your question here> [question5_end][Level5]
# ```
# 8.如果你的question中涉及到例子，你需要自己想象出一个例子（如请美化以下内容：[具体的内容案例]）

# [特定类别]：{category}
# [类别定义]：{define}
# 下面请你遵照给出的要求，生成属于[{category}]的5个question:
# """

# en_prompt = """
# As an expert in LLM evaluation question creation, your task is to create questions that humans might pose to the model. Based on a given "specific category", you must create Five questions, with varying levels of difficulty. The questions you create must meet the following criteria:
# 1. The questions must fit the given "specific category."
# 2. The questions must not plagiarize the "example question" and must be entirely different from it.
# 3. The questions should be as detailed and creative as possible, envisioning various scenarios and functions.
# 4. The five questions need to be as diverse as possible from one another.
# 5. The five questions should be posed at five different difficulty levels.
# 6. The questions must be reasonable and something a human would ask in that manner.
# 7. The questions must be replied to in the following format, which is to provide a difficulty level and description after each question:
# ```
# [question1_start] <Your question here> [question1_end][Level1]
# [question2_start] <Your question here> [question2_end][Level2]
# [question3_start] <Your question here> [question3_end][Level3]
# [question4_start] <Your question here> [question4_end][Level4]
# [question5_start] <Your question here> [question5_end][Level5]
# ```
# 8. If your question involves examples, you need to come up with your own example (for example, please beautify the following content: [specific content example]).

# [Specific Category]: {category}
# [Category Definition]: {define}
# Now, please generate five questions for the [{category}]:
# """

# zh_prompt = """
# 你是一个评测question制造专家，你需要制造出人类可能会对模型提出的问题，根据我给出的一个“特定类别”和“题目示例”来制造5个question，并且这5个question应该有难易区别。
# 你制造的question必须符合以下要求：
# 1.制造的question必须符合给出的“特定类别”
# 2.制造的question不能抄袭我的题目示例，必须要与之完全不同
# 3.制造的question应该尽量详细，并且具有创造性，能想象不同场景、不同功能的question
# 4.制造的5个question需要彼此之间尽量多样化
# 5.制造的5个question需要按从易到难的5个难度级别进行出题
# 6.制造的question必须合理，是人类口吻会问出的问题
# 7.制造的question要按照一定的格式进行回复，即在每个question后面需要给出它的难度评级:
# ```
# [question1_start] <Your question here> [question1_end][Level1]
# [question2_start] <Your question here> [question2_end][Level2]
# [question3_start] <Your question here> [question3_end][Level3]
# [question4_start] <Your question here> [question4_end][Level4]
# [question5_start] <Your question here> [question5_end][Level5]
# ```
# 8.如果你的question中涉及到例子，你需要自己想象出一个例子（如请美化以下内容：[具体的内容案例]）

# [特定类别]：{category}
# [类别定义]：{define}
# [题目示例]：{example}
# 下面请你遵照给出的要求，生成属于[{category}]的5个question:
# """

# en_prompt = """
# As an expert in LLM evaluation question creation, your task is to create questions that humans might pose to the model. Based on a given "specific category" and "example question", you must create Five questions, with varying levels of difficulty. The questions you create must meet the following criteria:
# 1. The questions must fit the given "specific category."
# 2. The questions must not plagiarize the "example question" and must be entirely different from it.
# 3. The questions should be as detailed and creative as possible, envisioning various scenarios and functions.
# 4. The five questions need to be as diverse as possible from one another.
# 5. The five questions should be posed at five different difficulty levels.
# 6. The questions must be reasonable and something a human would ask in that manner.
# 7. The questions must be replied to in the following format, which is to provide a difficulty level after each question:
# ```
# [question1_start] <Your question here> [question1_end][Level1]
# [question2_start] <Your question here> [question2_end][Level2]
# [question3_start] <Your question here> [question3_end][Level3]
# [question4_start] <Your question here> [question4_end][Level4]
# [question5_start] <Your question here> [question5_end][Level5]
# ```
# 8. If your question involves examples, you need to come up with your own example (for example, please beautify the following content: [specific content example]).

# [Specific Category]: {category}
# [Category Definition]: {define}
# [Example Question]: {example}
# Now, please generate five questions for the [{category}]:
# """

sub_map = {
    # "llm_tasks_w_def_full_zh": zh_prompt,
    # "llm_tasks_w_def_full_en": zh_prompt,
    'task_difficulties_zh': zh_prompt,
    # "task_difficulties_en": zh_prompt,
}

for _name, _prompt in sub_map.items():
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=_prompt),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer,
            max_seq_len=4096,
            max_out_len=2048,  # set as 1024 for yi models
        ),
    )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            infer_order='double',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role='HUMAN', prompt=_prompt),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            # type=TaskCategoriesDataset,
            type=TaskDifficultiesDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg))
