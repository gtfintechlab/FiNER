from enum import Enum, EnumMeta
from typing import Dict, List

entity_list: List[str] = ["PER", "LOC", "ORG"]

relationship_list: List[str] = [
    "CEO",
    "CTO",
    "COO",
    "CFO",
    "CMO",
    "CLO",
    "DIR",
    "CHAIRMAN",
    "PRESIDENT",
    "VICE_PRESIDENT",
    "SECRETARY",
    "TREASURER",
    "BOARD",

    "COMPETITOR",
    "MERGE",
    "SUPPLIER",
    "CUSTOMER",
    "LENDER",
    "AUDIT",
    "LAW",
    "SUBSIDIARY",
    "LAW_FIRM",

    "SITE_LOC",
    "HQ_LOC",

    "RISK",
    "TERMINATE",
    "HIRE"
]

# This is separate list to add suffixes e.g. _B to
# represent beginning of a multi-token entity
suffixes: List[str] = ["_B", "_I"]


def apply_suffixes(label_list: List[str]) -> List[str]:
    """

    :param label_list: List of labels.
    :return: List of labels in which each suffix is appended to each label in label_list
    """

    res_list: List[str] = []
    for label in label_list:
        for suffix in suffixes:
            res_list.append(label + suffix)

    return res_list


entities_with_suffix: List[str] = apply_suffixes(entity_list)
relationships_with_suffix: List[str] = apply_suffixes(relationship_list)

entities_dict: Dict[str, int] = dict(zip(entities_with_suffix, range(1, len(entities_with_suffix) + 1)))
relationships_dict: Dict[str, int] = dict(zip(relationships_with_suffix, range(1, len(relationships_with_suffix) + 1)))

entities_dict["O"] = 0
relationships_dict["O"] = 0


class Labels:
    ENTITIES: EnumMeta = Enum('ENTITIES', entities_dict)
    RELATIONSHIPS: EnumMeta = Enum('RELATIONSHIPS', relationships_dict)

    @staticmethod
    def get_entity_cardinality() -> int:
        return len(Labels.ENTITIES)

    @staticmethod
    def get_relationship_cardinality() -> int:
        return len(Labels.RELATIONSHIPS)
