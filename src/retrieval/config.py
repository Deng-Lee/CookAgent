# Centralized configuration for retrieval/generation thresholds and parameters.

# Retrieval result size.
TOP_K = 25
# Max number of parent candidates returned.
TOP_PARENTS = 5

# RRF hyperparameter for parent aggregation.
RRF_K = 60
# Minimum coverage ratio to consider a parent eligible.
COVERAGE_THRESHOLD = 0.5
# Weight of mean RRF score in overall_score.
RRF_WEIGHT = 0.35
# Weight of max chunk score in overall_score.
MAX_CHUNK_WEIGHT = 0.55
# Small bonus for coverage ratio to avoid over-penalizing long docs.
COVERAGE_WEIGHT = 0.1

# Minimum overall_score to treat top1 as valid.
T_MIN = 0.6
# Ambiguity trigger ratio: score2/score1 above this -> GOOD_BUT_AMBIGUOUS.
AMBIGUOUS_RATIO = 0.8

# Title chunk similarity required to treat top1 as a title hit.
DISH_TITLE_MIN = 0.8
# Minimum top1 chunk score for single-intent queries.
S_MIN = 0.75
# Minimum top1 chunk score for multi-intent (recommend) queries.
S_MIN_RECOMMEND = 0.68

# Intent confidence threshold for using layer1 evidence.
INTENT_MIN = 0.5

# Default number of candidates to return for multi-intent queries.
MULTI_INTENT_DEFAULT_COUNT = 3

# Excluded categories by default.
DEFAULT_EXCLUDED_CATEGORIES = {"示例菜", "调料", "半成品"}

# Synonyms that explicitly request excluded categories.
EXCLUDED_CATEGORY_SYNONYMS = {
    "示例菜": ["示例菜"],
    "调料": ["调料", "调味料", "调味品"],
    "半成品": ["半成品", "半成品菜"],
}

# Category synonyms used for detecting user intent.
CATEGORY_SYNONYMS = {
    "素菜": ["素", "素菜", "素食", "无肉"],
    "荤菜": ["荤", "荤菜", "肉", "肉菜", "有肉"],
    "水产": ["水产", "海鲜", "鱼", "虾", "蟹"],
    "汤": ["汤", "羹", "煲汤"],
    "甜点": ["甜点", "甜品", "点心"],
    "饮料": ["饮料", "饮品", "奶茶", "果汁"],
    "早餐": ["早餐", "早饭", "早点"],
    "主食": ["主食", "饭", "面", "粥", "饼"],
}

# Multi-intent query trigger keywords.
MULTI_INTENT_KEYWORDS = [
    "推荐",
    "几个",
    "几道",
    "几种",
    "一些",
    "多种",
    "有哪些",
    "推荐一下",
    "推荐点",
    "来点",
    "给点",
]

# Recipe-intent keywords (single dish how-to).
RECIPE_INTENT_KEYWORDS = ["怎么做", "做法", "步骤", "教程", "流程", "方法"]

# Rule-based intent score thresholds.
ASK_STEP_N_SCORE = 0.95
ASK_STEPS_SCORE = 0.7
ASK_INGREDIENTS_SCORE = 0.8
ASK_INGREDIENTS_QUANTITY_SCORE = 0.6
ASK_TIME_SCORE = 0.8
ASK_HEAT_SCORE = 0.8
ASK_SUBSTITUTION_SCORE = 0.7
ASK_TIPS_SCORE = 0.7
