import json
import os.path as osp
from textwrap import dedent

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

# from .subjective_cmp import SubjectiveCmpDataset
from ..base import BaseDataset


@LOAD_DATASET.register_module()
class TaskDifficultiesDataset(BaseDataset):

    def load(self, path: str, name: str):
        filename = osp.join(path, f'{name}.json')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                category = problem['category']
                definition = problem['definition']
                ref_or_not = problem['ref_or_not']
                level = problem['level']
                criteria = problem['criteria']
                example_instruction = problem['example_instruction']
                prompt = self.create_prompt_zh(category, definition,
                                               ref_or_not, criteria,
                                               example_instruction)

                others = {}
                raw_data.append({
                    'prompt': prompt,
                    'judge': {
                        'category': category,
                        'definition': definition,
                        'ref_or_not': ref_or_not,
                        'level': level,
                        'criteria': criteria,
                        'example_instruction': example_instruction,
                        'prompt': prompt,
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset

    def create_prompt_zh(
        self,
        category: str,
        definition: str,
        ref_or_not: int,
        criteria: str,
        example_instruction: str,
    ) -> str:

        num_questions = 3
        instruction_length = 600 if ref_or_not else 200
        ref_length = 500
        # category = '创意性写作'
        # definition = '生成具有创造性和独特性的文本内容'
        # category = '功能性写作'
        # definition = '模型生成具有特定目的和格式的文本内容，如电子邮件、工作周报、项目规划、法律文书、邀请帖等。'
        # category = '自然语言处理 - 文本总结'
        # definition = '提炼文本核心内容，生成简洁连贯的摘要。例如博客总结、文章总结、论文总结和笔记总结等。'
        # category = '重写 - 文本错误修改'
        # definition = '修正给定文本中的语法、拼写及用词错误，确保语义清晰准确。'

        # category = '{category}'
        # definition = '{definition}'

        ref_or_not = 0

        ref_requirement = f'7.你制造的每一个instruction必须自己想象一个参考文本，例如总结以下文章：[参考文本]' if ref_or_not else ''

        level_template = ''
        for i in range(1, num_questions + 1):
            level_template += f"""
            [instruction{i}_start] <Your {instruction_length} word instruction here> [instruction{i}_end]
            """

            if ref_or_not:
                level_template += f'[instruction{i}_reference_text_start] <Your {ref_length} words reference text here> [instruction{i}_reference_text_end]'

        zh_prompt = dedent(f"""
        你是一个LLM评测专家，你需要给大模型制造有挑战性的instruction，根据我给出的“特定类别”、“instruction制造标准”和“example instruction”来制造{num_questions}个instruction。你制造的instruction必须符合以下要求：

        1.制造的所有instruction必须符合给出的“特定类别”和“类别定义”
        2.制造的{num_questions}个instruction应该尽量详细，并且具有创造性，能想象不同场景的instruction
        3.制造的{num_questions}个instruction需要彼此之间具有多样化
        4.制造的所有instruction必须包含所有instruction制造标准项，且字数一定要符合要求
        5.制造的所有instruction不能提供可选答案
        6.制造的instruction要按照一定的格式进行回复，即把instruction写到对应的标识符之间:
        ```
        {level_template}
        ```
        {ref_requirement}

        [特定类别]：{category}
        [类别定义]：{definition}
        [instruction制造标准]：
        {criteria}
        [example instruction]：{example_instruction}
        下面请你遵照给出的要求，用中文生成属于[{category}]的{num_questions}个instruction和对应的参考文本:
        """)
        # print(zh_prompt)

        return zh_prompt

    def create_prompt_en(
        self,
        category: str,
        definition: str,
        ref_or_not: int,
        criteria: str,
        example_instruction: str,
    ) -> str:

        num_levels = 4
        num_criteria = 5
        num_instructions = 1
        instruction_length = 100
        ref_length = 150

        ref_or_not = 0

        ref_requirement = f"8.Based on the created example instruction, you must come up with yuor own reference text, for example: 'please summarize the following text: [reference text].'" if ref_or_not else ''

        level_template = ''
        for i in range(1, num_levels + 1):
            level_template += dedent(f"""
            [level{i}]
            [level{i}_criteria_start] <Your criteria here> [level{i}_criteria_end]
            [level{i}_example_instruction_start] <Your {i*instruction_length} word example instruction here> [level{i}_example_instruction_end]
            """)

            if ref_or_not:
                level_template += dedent(
                    f'[level{i}_reference_text_start] <Your {i*ref_length} word reference text here> [level{i}_reference_text_end]'
                )

        en_prompt = f"""
        You are an LLM evaluation expert, and you need to design difficulty criteria for large model instructions and create corresponding difficulty-level example instructions. Based on a given "specific category", create {num_levels} levels of difficulty instructions, with each level having {num_criteria} difficulty criteria and 1 example instruction. The difficulty criteria and example instructions you create must meet the following requirements:

        1. The created difficulty criteria and example instructions must comply with the given "specific category".
        2. The created difficulty criteria must be specific and quantifiable, provide a title for each standard.
        3. The created {num_levels} difficulty criteria need to be distinguishable from each other.
        4. The created {num_levels} difficulty criteria should be arranged from easy to hard, and the complexity increases as the difficulty level increases.
        5. The created example instruction must include all difficulty criteria items of the current level, and the word count cannot be less than {instruction_length} words.
        6. The created example instruction cannot provide optional answers.
        7. The created difficulty criteria and example instructions need to be replied in a certain format, that is, after each difficulty level, the corresponding {num_criteria} difficulty criteria, {num_instructions} example instruction and the involved reference text need to be given:
        ```
        {level_template}
        ```
        {ref_requirement}

        [Instruction category]: {category}
        [Category definition]: {definition}
        Now, please adhere to the given requirements and generate 4 instruction difficulty levels, corresponding 5 difficulty criteria and 1 example instruction for [{category}] in English:
        """

        return en_prompt
