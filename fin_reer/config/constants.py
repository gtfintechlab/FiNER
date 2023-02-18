from typing import Dict, Set


class PathConstants:
    EXAMPLE_FILE_PATH: str = "../data/text_files/1000232_0001558370-19-002105.txt"
    EXAMPLE_ENTITIES_TRAINING_DATA_SAVE_PATH: str = "../data/generated_training_data/example_entities_data.pickle"
    EXAMPLE_RELATIONSHIPS_TRAINING_DATA_SAVE_PATH: str = "../data/generated_training_data/example_relationships_data.pickle"

    MASTER_INPUT_TEXT_FILE = "../data/master_text.xlsx"
    PREFIX_MASTER_OUTPUT_TEXT_FILE_ENTITIES = "../data/reports/entities/"
    PREFIX_MASTER_OUTPUT_TEXT_FILE_RELATIONSHIPS = "../data/reports/relationships/"
    PREFIX_MASTER_OUTPUT_TEXT_FILE_ERROR = "../data/errors/"
    PREFIX_INPUT_TEXT_FILE_PATH = "../data/text_files/"
    PREFIX_DATA_SAVE_PATH = "../data/generated_training_data/"


class ModelConstants:
    VERBOSE: bool = True
    NUM_EPOCHS: int = 1000
    LOG_FREQ: int = 10
    SEED: int = 81


class UtilityConstants:
    LABEL_TYPE_ENTITY: str = "ENTITY"
    LABEL_TYPE_RELATIONSHIP: str = "RELATIONSHIP"

    BLACKLIST: Set[str] = {
        "A", "An", "The", "Chairman",
        "CEO", "President", "Director",
        "CTO", "CFO", "CMO", "COO", "CLO",
    }

    STR_TO_INT_LABEL_MAP: Dict[str, int] = {
        "O": 0,
        "O_B": 0,
        "B-O": 0,
        "O_I": 0,
        "I-O": 0,
        "PER_B": 1,
        "B-PER": 1,
        "PER_I": 2,
        "I-PER": 2,
        "LOC_B": 3,
        "B-LOC": 3,
        "LOC_I": 4,
        "I-LOC": 4,
        "ORG_B": 5,
        "B-ORG": 5,
        "ORG_I": 6,
        "I-ORG": 6,
        "GPE_B": 7,
        "B-GPE": 7,
        "GPE_I": 8,
        "I-GPE": 8,
        "PRODUCT_B": 9,
        "B-PRODUCT": 9,
        "PRODUCT_I": 10,
        "I-PRODUCT": 10,
        "LEGAL_B": 11,
        "B-LEGAL": 11,
        "LEGAL_I": 12,
        "I-LEGAL": 12,
        "MONEY_B": 13,
        "B-MONEY": 13,
        "MONEY_I": 14,
        "I-MONEY": 14,
        "GOV_B": 15,
        "B-GOV": 15,
        "GOV_I": 16,
        "I-GOV": 16,
        "PATENT_B": 17,
        "B-PATENT": 17,
        "PATENT_I": 18,
        "I-PATENT": 18,
        "CERTIFICATE_B": 19,
        "B-CERTIFICATE": 19,
        "CERTIFICATE_I": 20,
        "I-CERTIFICATE": 20,
        "COMPANY_B": 21,
        "B-COMPANY": 21,
        "COMPANY_I": 22,
        "I-COMPANY": 22,
        "LAWSUIT_B": 23,
        "B-LAWSUIT": 23,
        "LAWSUIT_I": 24,
        "I-LAWSUIT": 24,
        "ACT_B": 25,
        "B-ACT": 25,
        "ACT_I": 26,
        "I-ACT": 26,
        "GOVT_ENTITY_B": 27,
        "B-GOVT_ENTITY": 27,
        "GOVT_ENTITY_I": 28,
        "I-GOVT_ENTITY": 28,
        "AGREEMENT_B": 29,
        "B-AGREEMENT": 29,
        "AGREEMENT_I": 30,
        "I-AGREEMENT": 30,
        "DATE_B": 31,
        "B-DATE": 31,
        "DATE_I": 32,
        "I-DATE": 32,
        "PERCENT_B": 35,
        "B-PERCENT": 35,
        "PERCENT_I": 36,
        "I-PERCENT": 36,
        "GOVT_ORG_B": 39,
        "B-GOVT_ORG": 39,
        "GOVT_ORG_I": 40,
        "I-GOVT_ORG": 40,
        "CEO_B": 41,
        "B-CEO": 41,
        "CEO_I": 42,
        "I-CEO": 42,
        "CTO_B": 43,
        "B-CTO": 43,
        "CTO_I": 44,
        "I-CTO": 44,
        "COO_B": 45,
        "B-COO": 45,
        "COO_I": 46,
        "I-COO": 46,
        "CFO_B": 47,
        "B-CFO": 47,
        "CFO_I": 48,
        "I-CFO": 48,
        "CMO_B": 49,
        "B-CMO": 49,
        "CMO_I": 50,
        "I-CMO": 50,
        "CLO_B": 51,
        "B-CLO": 51,
        "CLO_I": 52,
        "I-CLO": 52,
        "DIR_B": 53,
        "B-DIR": 53,
        "DIR_I": 54,
        "I-DIR": 54,
        "CHAIRMAN_B": 55,
        "B-CHAIRMAN": 55,
        "CHAIRMAN_I": 56,
        "I-CHAIRMAN": 56,
        "PRESIDENT_B": 57,
        "B-PRESIDENT": 57,
        "PRESIDENT_I": 58,
        "I-PRESIDENT": 58,
        "VICE_PRESIDENT_B": 59,
        "B-VICE_PRESIDENT": 59,
        "VICE_PRESIDENT_I": 60,
        "I-VICE_PRESIDENT": 60,
        "SECRETARY_B": 61,
        "B-SECRETARY": 61,
        "SECRETARY_I": 62,
        "I-SECRETARY": 62,
        "TREASURER_B": 63,
        "B-TREASURER": 63,
        "TREASURER_I": 64,
        "I-TREASURER": 64,
        "BOARD_B": 65,
        "B-BOARD": 65,
        "BOARD_I": 66,
        "I-BOARD": 66,
        "COMPETITOR_B": 67,
        "B-COMPETITOR": 67,
        "COMPETITOR_I": 68,
        "I-COMPETITOR": 68,
        "MERGE_B": 69,
        "B-MERGE": 69,
        "MERGE_I": 70,
        "I-MERGE": 70,
        "SUPPLIER_B": 71,
        "B-SUPPLIER": 71,
        "SUPPLIER_I": 72,
        "I-SUPPLIER": 72,
        "CUSTOMER_B": 73,
        "B-CUSTOMER": 73,
        "CUSTOMER_I": 74,
        "I-CUSTOMER": 74,
        "LENDER_B": 75,
        "B-LENDER": 75,
        "LENDER_I": 76,
        "I-LENDER": 76,
        "AUDIT_B": 77,
        "B-AUDIT": 77,
        "AUDIT_I": 78,
        "I-AUDIT": 78,
        "LAW_B": 79,
        "B-LAW": 79,
        "LAW_I": 80,
        "I-LAW": 80,
        "SUBSIDIARY_B": 81,
        "B-SUBSIDIARY": 81,
        "SUBSIDIARY_I": 82,
        "I-SUBSIDIARY": 82,
        "LAW_FIRM_B": 83,
        "B-LAW_FIRM": 83,
        "LAW_FIRM_I": 84,
        "I-LAW_FIRM": 84,
        "SITE_LOC_B": 85,
        "B-SITE_LOC": 85,
        "SITE_LOC_I": 86,
        "I-SITE_LOC": 86,
        "HQ_LOC_B": 87,
        "B-HQ_LOC": 87,
        "HQ_LOC_I": 88,
        "I-HQ_LOC": 88,
        "RISK_B": 89,
        "B-RISK": 89,
        "RISK_I": 90,
        "I-RISK": 90,
        "TERMINATE_B": 91,
        "B-TERMINATE": 91,
        "TERMINATE_I": 92,
        "I-TERMINATE": 92,
        "HIRE_B": 93,
        "B-HIRE": 93,
        "HIRE_I": 94,
        "I-HIRE": 94,

        "B-AUTHOR": 0,
        "B-BOARD_MEMBER": 65,
        "B-BORROWER": 0,
        "B-CONSUMER": 73,
        "B-DIRECTOR": 53,
        "B-DIST_SITE": 0,
        "B-EDITOR": 0,
        "B-HQ_SITE": 87,
        "B-NEWS_AGENCY": 0,
        "B-OFFICE_SITE": 85,
        "B-PARENT_COMPANY": 0,
        "B-PROD_SITE": 85,
        "B-SECRETART": 61,
        "B-SUBSIDARY": 81,

        "I-AUTHOR": 0,
        "I-BOARD_MEMBER": 66,
        "I-BORROWER": 0,
        "I-CONSUMER": 74,
        "I-DIRECTOR": 54,
        "I-DIST_SITE": 0,
        "I-EDITOR": 0,
        "I-HQ_SITE": 88,
        "I-NEWS_AGENCY": 0,
        "I-OFFICE_SITE": 86,
        "I-PARENT_COMPANY": 0,
        "I-PROD_SITE": 86,
        "I-SECRETART": 62,
        "I-SUBSIDARY": 82
    }

    INT_TO_STR_MAP: Dict[int, str] = {
        -1: "O",
        0: "O",
        1: "PER",
        2: "PER",
        3: "LOC",
        4: "LOC",
        5: "ORG",
        6: "ORG",
        7: "GPE",
        8: "GPE",
        9: "PRODUCT",
        10: "PRODUCT",
        11: "LEGAL",
        12: "LEGAL",
        13: "MONEY",
        14: "MONEY",
        15: "GOV",
        16: "GOV",
        17: "PATENT",
        18: "PATENT",
        19: "CERTIFICATE",
        20: "CERTIFICATE",
        21: "COMPANY",
        22: "COMPANY",
        23: "LAWSUIT",
        24: "LAWSUIT",
        25: "ACT",
        26: "ACT",
        27: "GOVT_ENTITY",
        28: "GOVT_ENTITY",
        29: "AGREEMENT",
        30: "AGREEMENT",
        31: "DATE",
        32: "DATE",
        35: "PERCENT",
        36: "PERCENT",
        39: "GOVT_ORG",
        40: "GOVT_ORG",
        41: "CEO",
        42: "CEO",
        43: "CTO",
        44: "CTO",
        45: "COO",
        46: "COO",
        47: "CFO",
        48: "CFO",
        49: "CMO",
        50: "CMO",
        51: "CLO",
        52: "CLO",
        53: "DIR",
        54: "DIR",
        55: "CHAIRMAN",
        56: "CHAIRMAN",
        57: "PRESIDENT",
        58: "PRESIDENT",
        59: "VICE_PRESIDENT",
        60: "VICE_PRESIDENT",
        61: "SECRETARY",
        62: "SECRETARY",
        63: "TREASURER",
        64: "TREASURER",
        65: "BOARD",
        66: "BOARD",
        67: "COMPETITOR",
        68: "COMPETITOR",
        69: "MERGE",
        70: "MERGE",
        71: "SUPPLIER",
        72: "SUPPLIER",
        73: "CUSTOMER",
        74: "CUSTOMER",
        75: "LENDER",
        76: "LENDER",
        77: "AUDIT",
        78: "AUDIT",
        79: "LAW",
        80: "LAW",
        81: "SUBSIDIARY",
        82: "SUBSIDIARY",
        83: "LAW_FIRM",
        84: "LAW_FIRM",
        85: "SITE_LOC",
        86: "SITE_LOC",
        87: "HQ_LOC",
        88: "HQ_LOC",
        89: "RISK",
        90: "RISK",
        91: "TERMINATE",
        92: "TERMINATE",
        93: "HIRE",
        94: "HIRE",
    }

    POPULAR_NAMES: Set[str] = {'James', 'Robert', 'John', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles',
             'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua',
             'Kenneth', 'Kevin', 'Brian', 'George', 'Edward', 'Ronald', 'Timothy', 'Jason', 'Jeffrey', 'Ryan', 'Jacob',
             'Gary', 'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon', 'Benjamin',
             'Samuel', 'Gregory', 'Frank', 'Alexander', 'Raymond', 'Patrick', 'Jack', 'Dennis', 'Jerry', 'Tyler',
             'Aaron', 'Jose', 'Adam', 'Henry', 'Nathan', 'Douglas', 'Zachary', 'Peter', 'Kyle', 'Walter', 'Ethan',
             'Jeremy', 'Harold', 'Keith', 'Christian', 'Roger', 'Noah', 'Gerald', 'Carl', 'Terry', 'Sean', 'Austin',
             'Arthur', 'Lawrence', 'Jesse', 'Dylan', 'Bryan', 'Joe', 'Jordan', 'Billy', 'Bruce', 'Albert', 'Willie',
             'Gabriel', 'Logan', 'Alan', 'Juan', 'Wayne', 'Roy', 'Ralph', 'Randy', 'Eugene', 'Vincent', 'Russell',
             'Elijah', 'Louis', 'Bobby', 'Philip', 'Johnny', 'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth',
             'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Margaret', 'Sandra', 'Ashley',
             'Kimberly', 'Emily', 'Donna', 'Michelle', 'Dorothy', 'Carol', 'Amanda', 'Melissa', 'Deborah', 'Stephanie',
             'Rebecca', 'Sharon', 'Laura', 'Cynthia', 'Kathleen', 'Amy', 'Shirley', 'Angela', 'Helen', 'Anna', 'Brenda',
             'Pamela', 'Nicole', 'Emma', 'Samantha', 'Katherine', 'Christine', 'Debra', 'Rachel', 'Catherine',
             'Carolyn', 'Janet', 'Ruth', 'Maria', 'Heather', 'Diane', 'Virginia', 'Julie', 'Joyce', 'Victoria',
             'Olivia', 'Kelly', 'Christina', 'Lauren', 'Joan', 'Evelyn', 'Judith', 'Megan', 'Cheryl', 'Andrea',
             'Hannah', 'Martha', 'Jacqueline', 'Frances', 'Gloria', 'Ann', 'Teresa', 'Kathryn', 'Sara', 'Janice',
             'Jean', 'Alice', 'Madison', 'Doris', 'Abigail', 'Julia', 'Judy', 'Grace', 'Denise', 'Amber', 'Marilyn',
             'Beverly', 'Danielle', 'Theresa', 'Sophia', 'Marie', 'Diana', 'Brittany', 'Natalie', 'Isabella',
             'Charlotte', 'Rose', 'Alexis', 'Kayla'}