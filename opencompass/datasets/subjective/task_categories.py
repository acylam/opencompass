import json
import os.path as osp
from textwrap import dedent

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

# from .subjective_cmp import SubjectiveCmpDataset
from ..base import BaseDataset

# from typing import Dict


@LOAD_DATASET.register_module()
class TaskCategoriesDataset(BaseDataset):

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
                ref_instruction = problem['reference_instruction']
                if 'llm_tasks_w_def_full_zh' in name:
                    prompt = self.create_prompt_zh(category, definition,
                                                   ref_or_not, ref_instruction)
                else:
                    prompt = self.create_prompt_en(category, definition,
                                                   ref_or_not, ref_instruction)
                # others = problem['others']
                raw_data.append({
                    'prompt': prompt,
                    'judge': {
                        'definition': definition,
                        'category': category,
                        'ref_or_not': ref_or_not,
                        'prompt': prompt,
                        'ref_instruction': ref_instruction,
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset

    def create_prompt_zh(self, category: str, definition: str, ref_or_not: int,
                         ref_instruction: str) -> str:

        num_levels = 4
        num_criteria = 5
        num_instructions = 1
        instruction_length = 100 if ref_instruction == '' else 200
        ref_length = 150
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

        ref_requirement = f'8.你必须根据制造的example instruction自己想象一个reference text原文，例如总结以下文章：[具体需要总结的原文]' if ref_or_not else ''

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

        ref_inst_requirement = f'8.制造的example instruction不能抄袭我提供的[题目示例]，必须要与之完全不同' if ref_instruction != '' else ''

        ref_instruction = '[题目示例]：\n' + ref_instruction if ref_instruction != '' else ''
        gen_ref_instruction = '和“题目示例”' if ref_instruction != '' else ''

        zh_prompt = f"""
        你是一个LLM评测专家，你需要设计大模型instruction的难度标准和制造对应难度的example instruction，根据我给出的一个“特定类别”{gen_ref_instruction}来制造{num_levels}个instruction难度级别，每个级别设计{num_criteria}个难度标准和{num_instructions}个example instruction。你制造的难度标准和example instruction必须符合以下要求：

        1.制造的难度标准和example instruction必须符合给出的“特定类别”
        2.制造的难度标准要具体和可量化，给每个标准提供一个标题
        3.制造的{num_levels}个难度标准需要彼此之间具有区分度
        4.制造的{num_levels}个难度标准需要按从易到难进行出题，随着难度等级上升而提高复杂度
        5.制造的example instruction一定要包含当前等级的所有难度标准项，且字数不能少于{instruction_length}字
        6.制造的example instruction不能提供可选答案
        7.制造的难度标准和example instruction要按照一定的格式进行回复，即在每个难度级别后面需要给出对应的{num_criteria}个难度标准、{num_instructions}个example instruction和涉及的reference text:
        ```
        {level_template}
        ```
        {ref_requirement}
        {ref_inst_requirement}

        [特定类别]：{category}
        [类别定义]：{definition}
        {ref_instruction}

        下面请你遵照给出的要求，用中文生成属于[{category}]的{num_levels}个instruction难度级别、对应的{num_criteria}个难度标准和{num_instructions}个example instruction:
        """
        # print(zh_prompt)

        return zh_prompt

    def create_prompt_en(self, category: str, definition: str, ref_or_not: int,
                         ref_instruction: str) -> str:

        num_levels = 4
        num_criteria = 5
        num_instructions = 1
        instruction_length = 100 if ref_instruction == '' else 200
        ref_length = 150

        ref_or_not = 0

        ref_requirement = f"8.Based on the created example instruction, you must come up with your own reference text, for example: 'please summarize the following text: [reference text].'" if ref_or_not else ''

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

        ref_inst_requirement = f'8.Your example instruction should have a similar format and style as the [Sample instruction] WITHOUT plagiarizing its content.' if ref_instruction != '' else ''

        ref_instruction = '[Sample instruction]：\n' + ref_instruction if ref_instruction != '' else ''
        gen_ref_instruction = ' and "Sample instruction"' if ref_instruction != '' else ''

        en_prompt = f"""
        You are an LLM evaluation expert, and you need to design difficulty criteria for LLM instructions and create corresponding example instructions for each difficulty level. Based on a given "Instruction category"{gen_ref_instruction}, create {num_levels} instruction difficulty levels, with each level having {num_criteria} difficulty criteria and 1 example instruction. The difficulty criteria and example instructions you create must meet the following requirements:

        1. Your difficulty criteria and example instructions must comply with the given "specific category".
        2. Your difficulty criteria must be specific and quantifiable, provide a title for each standard.
        3. Your {num_levels} difficulty criteria need to be distinguishable from each other.
        4. Your {num_levels} difficulty criteria should be arranged from easy to hard, and the complexity increases as the difficulty level increases.
        5. Your example instruction must include all difficulty criteria items of the current level, and the word count cannot be less than {instruction_length} words.
        6. Your example instruction cannot provide multiple choice answers.
        7. Your difficulty criteria and example instructions need to be replied in a certain format, that is, after each difficulty level, the corresponding {num_criteria} difficulty criteria, {num_instructions} example instruction and the involved reference text need to be given:
        ```
        {level_template}
        ```
        {ref_requirement}
        {ref_inst_requirement}

        [Instruction category]: {category}
        [Category definition]: {definition}
        {ref_instruction}

        Now, please adhere to the given requirements and generate {num_levels} instruction difficulty levels, their corresponding {num_criteria} difficulty criteria and {num_instructions} example instruction for [{category}] in English:
        """

        return en_prompt
