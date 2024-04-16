import json
from pathlib import Path
from typing import Any, Union
from functools import reduce
from operator import getitem
from collections import defaultdict


def set_nested_item(
    input_dict: dict, 
    map_list: list, 
    value: Any
):
    """Set item in nested dictionary"""
    reduce(getitem, map_list[:-1], input_dict)[map_list[-1]] = value
    return input_dict


def dd_rec():
    return defaultdict(dd_rec)


def defaultify(d):
    if not isinstance(d, dict):
        return d
    return defaultdict(dd_rec, {k: defaultify(v) for k, v in d.items()})


def json_attr_injector(
    from_json_file: Union[str, Path],
    dest_json_file: Union[str, Path],
    output_json_dir: Union[str, Path],
    attribute: str,
    inject_into: list[str],
    datasets: list[str],
    output_file_suffix: str = '',
) -> None:
    
    if isinstance(from_json_file, str):
        from_json_file = Path(from_json_file)

    if isinstance(dest_json_file, str):
        dest_json_file = Path(dest_json_file)

    if isinstance(output_json_dir, str):
        output_json_dir = Path(output_json_dir)

    if isinstance(inject_into, str):
        inject_into = list(inject_into)

    if isinstance(datasets, str):
        datasets = list(datasets)

    for dataset in datasets:
        with open(from_json_file / f'{dataset}.json', encoding='utf-8') as f:
            from_dict = json.load(f)

        with open(dest_json_file / f'{dataset}.json', encoding='utf-8') as f:
            dest_list = json.load(f)

        for key, value in from_dict.items():
            # print(key)

            dest_list[int(key)] = defaultify(dest_list[int(key)])

            dest_list[int(key)] = set_nested_item(dest_list[int(key)], inject_into, value[attribute])

        with open(output_json_dir / f'{dataset}{output_file_suffix}.json', 'w', encoding='utf-8') as f:
            json.dump(dest_list, f, indent=4, ensure_ascii = False)


if __name__ == "__main__":
    # Inject guidelines into others.guideline location of creationv2_zh
    base_path = Path.home() / ('projects/opencompass/data/subjective/compass_arena/')
    
    json_attr_injector(
        from_json_file=base_path / 'guidelines',
        dest_json_file=base_path,
        output_json_dir=base_path,
        output_file_suffix='_joined',
        attribute = 'prediction',
        inject_into = ['others', 'guideline'],
        datasets=[
            'creationv2_zh',
            'creationv2_en',
        ],
    )
