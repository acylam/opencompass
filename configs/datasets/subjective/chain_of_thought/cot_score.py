from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassArenaDataset, ChainOfThoughtJudgeDataset


subjective_reader_cfg = dict(
    input_columns=['question', 'ref', 'guideline'],
    output_column='judge',
    )

data_path = "data/subjective/compass_arena/"
# data_path = "/mnt/petrelfs/share_data/caomaosong/subjective/compass_arena"

subjective_datasets = []

creation_prompt_zh = """
请根据提供的 用户问题，评分要求 以及 相应的两个回答（回答1，回答2），判断两个回答中哪一个更好。你必须按照评分要求进行选择。

[用户问题]
{question}

[评分要求]
{guideline}

[回答1开始]
{prediction}
[回答1结束]

[回答2开始]
{prediction2}
[回答2结束]

根据评分要求，在以下 3 个选项中做出选择:
A. 回答1更好
B. 回答2更好
C. 回答1、2平局
并提供你的解释原因。

如果你认为回答1更好，你的输出应形如：
选择：A
原因：blahblah blahblah\n

如果你认为回答2更好，你的输出应形如：
选择：B
原因：blahblah blahblah\n

如果你认为回答1、2打成平手，你的输出应形如：
选择：C
原因：blahblah blahblah\n
"""


creation_prompt_en = """
Please use the provided user question and evaluation criteria to determine which of the two answers (Answer 1, Answer 2) is better. You must adhere to the evaluation criteria provided.

[User Question]
{question}

[Evaluation Criteria]
{guideline}

[Start of Answer 1]
{prediction}
[End of Answer 1]

[Start of Answer 2]
{prediction2}
[End of Answer 2]

Based on the evaluation criteria, make a choice among the following 3 options and provide your reasoning:
A. Answer 1 is better.
B. Answer 2 is better.
C. Answer 1 and 2 are tied.

If you believe Answer 1 is better, your output should be in the form of:
Choice: A
Reason: blahblah blahblah\n

If you believe Answer 2 is better, your output should be in the form of:
Choice: B
Reason: blahblah blahblah\n

If you believe Answer 1 and 2 are tied, your output should be in the form of:
Choice: C
Reason: blahblah blahblah\n
"""

# """
# [参考用户问题开始]
# 一个人乐意去探索陌生世界，仅仅是因为好奇心吗？请写一篇文章，谈谈你对这个问题的认识和思考。要求：(1） 自拟题目；（2）不少于 800字。
# [参考用户问题结束]

# [参考评分标准开始]
# 评分标准（重要性依次递减）:
# 1. 好的回答必须首先符合用户问题里的所有需求，不能有遗漏，且不能跑题。
# 2. 好的回答必须具有逻辑连贯性，并围绕一个中心进行回答。
# 3. 好的回答必须具有创造性的词语和表达丰富度。
# [参考评分标准结束]
# """

# dynamic_creative_judge_prompt = """
# 你是一个公正的语言模型评测员，请根据你对以下 用户问题 的理解，自己制定相关的 评分标准 来评价 AI助手模型的回答，并根据制定好的 评分标准 进行评价和打分。

# [用户问题开始]
# {question}
# [用户问题结束]
# """ + base_prompt

def list_prompt_template(
    tag: str, 
    key: str, 
    value: str, 
    n: int,
    suffix: str = None,
) -> str:
    prompt = f"<{tag}Start>\n"
    for i in range(1, n + 1):
        prompt += f"{i}. {key}{i}：{value}\n"

    if suffix is not None:
        prompt += suffix

    prompt += f"<{tag}End>\n"
    return prompt

ref_guidlines_prompt = list_prompt_template(
    tag='Guideline', 
    key='标准', 
    value='blahblahblah', 
    n=5
)

ref_response_prompt = list_prompt_template(
    tag='Response', 
    key='标准', 
    value='blahblahblah', 
    n=5
)

ref_score_prompt = list_prompt_template(
    tag='Score', 
    key='标准', 
    value='评分', 
    suffix='\n最终评分：评分\n',
    n=5
)

# """
# 你设计的评估标准必须包含以下几项要求，但可以根据问题的内容和要求进行补充：文章的原创性、上下文连贯性、语法正确性、用词合理性、字数和格式要求。

# 针对<QuestionStart><QustionEnd>之间的问题，根据在<GuidelineStart><GuidelineEnd>之间定义的评估标准，对<AnswerStart><AnswerEnd>之间的回复进行评分，分数在<ScoreStart><ScoreEnd>之间，满分10分。

# 现在你要扮演一个人类专家，对大模型模型在具体问题上的回复进行评价，请针对<QuestionStart><QustionEnd>之间的问题和<GuidelineStart><GuidelineEnd>之间的评估标准，对<AnswerStart><AnswerEnd>之间的回复进行评分。

# 评分依据定义在<ResponseStart><ResponseEnd>之间。最终评分定义在<ScoreStart><ScoreEnd>之间，满分10分。

# 现在你要扮演一个人类专家，对大模型在具体问题上的回复进行评价，请针对<QuestionStart><QustionEnd>之间的问题和<GuidelineStart><GuidelineEnd>之间的评估标准，对<AnswerStart><AnswerEnd>之间的回复进行评分，评分依据定义在<ResponseStart><ResponseEnd>之间。最终评分满分10分，定义在<ScoreStart><ScoreEnd>之间。

# 现在你要扮演一个人类专家，对大模型在具体问题上的回复进行评价，请针对<QuestionStart><QustionEnd>之间的问题和<GuidelineStart><GuidelineEnd>之间的评价标准，对<AnswerStart><AnswerEnd>之间的回复进行评价，评价定义在<ResponseStart><ResponseEnd>之间。最终评分定义在<ScoreStart><ScoreEnd>之间，满分10分。

# """

# dynamic_creative_judge_prompt = f"""
# 现在你要扮演一个人类专家，对大模型在具体问题上的回复进行评价，请针对<QuestionStart><QustionEnd>之间的问题和<GuidelineStart><GuidelineEnd>之间的评价标准对<AnswerStart><AnswerEnd>之间的回复进行评价，并完成以下2项任务：

# 1. 评价定义在<ResponseStart><ResponseEnd>之间。

# 2. 最终评分（满分10分）定义在<ScoreStart><ScoreEnd>之间。

# <QuestionStart>
# {{question}}
# <QustionEnd>

# {{guideline}}

# <AnswerStart>
# {{prediction}}
# <AnswerEnd>

# <ResponseStart>
# [标准1]：blahblahblah
# [标准2]：blahblahblah
# [标准3]：blahblahblah
# [标准4]：blahblahblah
# [标准5]：blahblahblah
# <ResponseEnd>

# <ScoreStart>
# 最终评分：[评分]分
# <ScoreEnd>
# """

# dynamic_creative_judge_prompt = f"""
# 现在你要扮演一个人类专家，对大模型在具体问题上的回复进行评价，请针对<GuidelineStart><GuidelineEnd>之间的评价标准对<AnswerStart><AnswerEnd>之间的答案进行评价。评价依据定义在<ResponseStart><ResponseEnd>之间。最终评分（满分10分）定义在<ScoreStart><ScoreEnd>之间。

# {{guideline}}

# <AnswerStart>
# {{prediction}}
# <AnswerEnd>

# <ScoreStart>
# 最终评分：[评分]
# <ScoreEnd>
# """

dynamic_creative_judge_prompt_zh = f"""
请根据提供的 用户问题，评分要求 以及 相应的回答，给出评分。你必须按照评分要求进行打分。

[用户问题]
{{question}}

[评分要求]
{{guideline}}

[回答开始]
{{prediction}}
[回答结束]

根据评分要求，从10分满分开始，对回答中*每一项*不符合要求的内容*扣1分*，直到扣完为止。你的输出应形如：
最终评分：[评分]
原因：blahblahblah\n
"""

dynamic_creative_judge_prompt_en = f"""
Please use the provided user question and evaluation criteria to score the corresponding answer. You must adhere to the evaluation criteria provided.

[User Question]
{{question}}

[Evaluation Criteria]
{{guideline}}

[Start of Answer]
{{prediction}}
[End of Answer]

Based on the evaluation criteria, evaluate the answer by deducting scores from the maximum of 10.00 for any part of the answer that does not meet the criteria. Provide your reasoning. Your output should be in the form of:
Score: [score]
Reason: blahblah blahblah\n
"""

sub_map = dict(
    # creationv2_zh=creation_prompt_zh,
    # creationv2_en=creation_prompt_en,
    creationv2_zh=dynamic_creative_judge_prompt_zh,
    # creationv2_en=dynamic_creative_judge_prompt_en,
)

for _name, _prompt in sub_map.items():
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt="{question}"
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_seq_len=4096, max_out_len=2048),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            # infer_order='double',
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
            # type=CompassArenaDataset,
            type=ChainOfThoughtJudgeDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
