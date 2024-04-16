from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .subjective_cmp import SubjectiveCmpDataset


@LOAD_DATASET.register_module()
class ChainOfThoughtJudgeDataset(SubjectiveCmpDataset):

    def load(
        self,
        path: str,
        name: str,
    ):
        dataset = list(super().load(path, name))
        creation_dataset = []
        for data in dataset:
            data['ref'] = '满足用户需求，言之有理即可'
            data['guideline'] = 'guideline not available'

            if 'reference' in data['others']:
                data['ref'] = '满足用户需求，言之有理即可'

                if data['others']['reference'] is not None:
                    data['ref'] = data['others']['reference']
                    
            if 'guideline' in data['others']:
                data['guideline'] = 'guideline not available'

                if data['others']['guideline'] is not None:
                    data['guideline'] = data['others']['guideline']

            creation_dataset.append(data)
            
        dataset = Dataset.from_list(creation_dataset)

        return dataset
