"""Rule-based parser for verifier-side causal claim analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from verifier.outputs import ClaimPolarity, ClaimStrength, ParsedClaim, QueryType


_VARIABLE_TOKEN = r"[A-Za-z][A-Za-z0-9_]*"
_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "answer",
    "available",
    "causal",
    "claim",
    "conclusion",
    "counterfactual",
    "current",
    "data",
    "different",
    "effect",
    "evidence",
    "identified",
    "individual",
    "observed",
    "same",
    "sample",
    "the",
    "through",
    "together",
    "under",
    "unit",
    "what",
}

_CHINESE_CAUSAL_VERBS = r"导致|造成|影响|改变|驱动|决定|作用于|识别"
_CHINESE_ASSOCIATION_TERMS = r"相关性|关联|相关|关系"
_CHINESE_TENTATIVE_TERMS = r"可能|也许|或许|似乎|看起来"

_QUERY_PATTERNS: dict[QueryType, tuple[tuple[re.Pattern[str], int], ...]] = {
    QueryType.ASSOCIATION: (
        (re.compile(r"\bassociation\b", re.IGNORECASE), 5),
        (re.compile(r"\bcorrelation\b", re.IGNORECASE), 5),
        (re.compile(r"\bobserved relationship\b", re.IGNORECASE), 4),
        (re.compile(r"\bobserved variables?\b", re.IGNORECASE), 3),
        (re.compile(r"\bmeasured variables?\b", re.IGNORECASE), 3),
        (re.compile(r"\bmove together\b", re.IGNORECASE), 4),
        (re.compile(r"\bselected sample\b", re.IGNORECASE), 4),
        (re.compile(r"\bselection\b", re.IGNORECASE), 4),
        (re.compile(r"\bcollider\b", re.IGNORECASE), 4),
        (re.compile(r"\bhidden[- ]variable\b", re.IGNORECASE), 4),
        (re.compile(r"\bomitted variable\b", re.IGNORECASE), 4),
        (re.compile(r"\blatent confounding\b", re.IGNORECASE), 4),
        (re.compile(r"相关性"), 5),
        (re.compile(r"关联"), 4),
        (re.compile(r"一起变化|共同变化"), 4),
        (re.compile(r"选择偏差|选择机制"), 4),
        (re.compile(r"隐藏变量|未观察混杂|遗漏变量"), 4),
        (re.compile(r"\bpattern\b", re.IGNORECASE), 1),
    ),
    QueryType.INTERVENTION: (
        (re.compile(r"\bintervene(?: on)?\b", re.IGNORECASE), 5),
        (re.compile(r"\bdo\(", re.IGNORECASE), 5),
        (re.compile(r"\bcausal effect\b", re.IGNORECASE), 5),
        (re.compile(r"\baverage treatment effect\b", re.IGNORECASE), 5),
        (re.compile(r"\binterventional effect\b", re.IGNORECASE), 5),
        (re.compile(r"\baverage_treatment_effect\b", re.IGNORECASE), 5),
        (re.compile(r"\binterventional_effect\b", re.IGNORECASE), 5),
        (re.compile(r"\beffect of\b", re.IGNORECASE), 4),
        (re.compile(r"\btreatment effect\b", re.IGNORECASE), 4),
        (re.compile(r"\badjust(?:ing|ment)?\b", re.IGNORECASE), 4),
        (re.compile(r"\bcontrol(?:ling)? for\b", re.IGNORECASE), 4),
        (re.compile(r"\bbackdoor\b", re.IGNORECASE), 4),
        (re.compile(r"\binstrument(?:al variable)?\b", re.IGNORECASE), 5),
        (re.compile(r"\biv\b", re.IGNORECASE), 4),
        (re.compile(r"\bexclusion restriction\b", re.IGNORECASE), 4),
        (re.compile(r"\bidentified\b", re.IGNORECASE), 2),
        (re.compile(r"干预|因果效应|工具变量|后门|前门|中介"), 5),
        (re.compile(rf"{_CHINESE_CAUSAL_VERBS}"), 3),
    ),
    QueryType.COUNTERFACTUAL: (
        (re.compile(r"\bcounterfactual\b", re.IGNORECASE), 5),
        (re.compile(r"\bcounterfactual response\b", re.IGNORECASE), 5),
        (re.compile(r"\bunit[- ]level\b", re.IGNORECASE), 4),
        (re.compile(r"\bunit_level_counterfactual\b", re.IGNORECASE), 5),
        (re.compile(r"\bwould have been\b", re.IGNORECASE), 4),
        (re.compile(r"\bwould definitely\b", re.IGNORECASE), 4),
        (re.compile(r"\bfor an individual\b", re.IGNORECASE), 4),
        (re.compile(r"\bsame observed history\b", re.IGNORECASE), 5),
        (re.compile(r"\bfor the same case\b", re.IGNORECASE), 5),
        (re.compile(r"\bunder a different value of\b", re.IGNORECASE), 4),
        (re.compile(r"\bswitching\b", re.IGNORECASE), 3),
        (re.compile(r"\beffect of treatment on the treated\b", re.IGNORECASE), 5),
        (re.compile(r"\beffect_of_treatment_on_treated\b", re.IGNORECASE), 5),
        (re.compile(r"\babduction[- ]action\b", re.IGNORECASE), 4),
        (re.compile(r"\babduction_action_prediction\b", re.IGNORECASE), 4),
        (re.compile(r"反事实"), 5),
        (re.compile(r"同一观测历史|同一个个体|同一个案例"), 5),
        (re.compile(r"如果把|换成另一个值"), 4),
    ),
}

_PAIR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b(?:causal|counterfactual)?\s*effect of (?:changing |switching |intervening on )?"
        rf"(?P<treatment>{_VARIABLE_TOKEN}) on (?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\busing {_VARIABLE_TOKEN} as an instrument is enough to recover the causal (?:impact|effect) of "
        rf"(?P<treatment>{_VARIABLE_TOKEN}) on (?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bthe variation induced by {_VARIABLE_TOKEN} isolates the true effect of "
        rf"(?P<treatment>{_VARIABLE_TOKEN}) on (?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bafter controlling for {_VARIABLE_TOKEN}, the causal effect of "
        rf"(?P<treatment>{_VARIABLE_TOKEN}) on (?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bcausal estimate from (?P<treatment>{_VARIABLE_TOKEN}) to "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:causal impact|impact|link|path) of (?P<treatment>{_VARIABLE_TOKEN}) "
        rf"on (?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bthe answer about (?P<treatment>{_VARIABLE_TOKEN}) and "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bthe consistent association between (?P<treatment>{_VARIABLE_TOKEN}) and "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bonce (?P<treatment>{_VARIABLE_TOKEN}) and (?P<outcome>{_VARIABLE_TOKEN}) move together\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\blatent confounding for (?P<treatment>{_VARIABLE_TOKEN}) and "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<treatment>{_VARIABLE_TOKEN}) itself is what drives "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bthe conclusion that (?P<treatment>{_VARIABLE_TOKEN}) affects "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_VARIABLE_TOKEN} affects (?P<outcome>{_VARIABLE_TOKEN}) only through "
        rf"(?P<treatment>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bpath from {_VARIABLE_TOKEN} to (?P<outcome>{_VARIABLE_TOKEN}) except through "
        rf"(?P<treatment>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bonly shifts (?P<treatment>{_VARIABLE_TOKEN}), it gives a valid causal estimate for "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bwhat (?P<outcome>{_VARIABLE_TOKEN}) would have been under a different value of "
        rf"(?P<treatment>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bcounterfactual prediction for (?P<outcome>{_VARIABLE_TOKEN}) under a different value of "
        rf"(?P<treatment>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\babduction[- ]action style counterfactual prediction for (?P<outcome>{_VARIABLE_TOKEN}) under a different value of "
        rf"(?P<treatment>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bswitching (?P<treatment>{_VARIABLE_TOKEN})\b.*?\bchange "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bchanging (?P<treatment>{_VARIABLE_TOKEN})\b.*?\bchange "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bcounterfactual response of (?P<outcome>{_VARIABLE_TOKEN}) to "
        rf"(?P<treatment>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bcounterfactual effect of treatment on the treated for (?P<treatment>{_VARIABLE_TOKEN}) on "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\beffect of treatment on the treated for (?P<treatment>{_VARIABLE_TOKEN}) on "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bhow (?P<outcome>{_VARIABLE_TOKEN}) would change under a different "
        rf"(?P<treatment>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bmechanism linking (?P<treatment>{_VARIABLE_TOKEN}), {_VARIABLE_TOKEN}, and "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_VARIABLE_TOKEN} is the only adjustment needed to justify a causal estimate from "
        rf"(?P<treatment>{_VARIABLE_TOKEN}) to (?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bsupports (?P<treatment>{_VARIABLE_TOKEN}) as a cause of "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<treatment>{_VARIABLE_TOKEN})\s+(?:itself\s+|directly\s+|clearly\s+|really\s+|only\s+|definitely\s+|probably\s+)?"
        rf"(?:appears?\s+to\s+|seems?\s+to\s+|may\s+|might\s+|could\s+)?"
        rf"(?:causes?|affects?|drives?|determines?|changes?|influences?|identifies?)\s+"
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bassociation between (?P<treatment>{_VARIABLE_TOKEN}) and "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\brelationship between (?P<treatment>{_VARIABLE_TOKEN}) and "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\blink between (?P<treatment>{_VARIABLE_TOKEN}) and "
        rf"(?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\blink from (?P<treatment>{_VARIABLE_TOKEN}) to (?P<outcome>{_VARIABLE_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<treatment>{_VARIABLE_TOKEN})\s*(?:本身\s*)?(?:直接\s*|明确地\s*|清楚地\s*|确实\s*|真的\s*)?"
        rf"(?:{_CHINESE_TENTATIVE_TERMS}\s*)?(?:{_CHINESE_CAUSAL_VERBS})\s*(?P<outcome>{_VARIABLE_TOKEN})"
    ),
    re.compile(
        rf"(?P<treatment>{_VARIABLE_TOKEN})\s*对\s*(?P<outcome>{_VARIABLE_TOKEN})\s*的(?:干预|因果)?(?:效应|影响)"
    ),
    re.compile(
        rf"(?P<treatment>{_VARIABLE_TOKEN})\s*与\s*(?P<outcome>{_VARIABLE_TOKEN})\s*的(?:{_CHINESE_ASSOCIATION_TERMS})"
    ),
    re.compile(
        rf"用于识别\s*(?P<treatment>{_VARIABLE_TOKEN})\s*[-=]>\s*(?P<outcome>{_VARIABLE_TOKEN})\s*的工具变量"
    ),
)

_NEGATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bno causal effect\b", re.IGNORECASE),
    re.compile(r"\bdoes not (?:cause|affect|change|determine|identify)\b", re.IGNORECASE),
    re.compile(r"\bdoesn't (?:cause|affect|change|determine|identify)\b", re.IGNORECASE),
    re.compile(r"\bcannot (?:identify|determine|support)\b", re.IGNORECASE),
    re.compile(r"\bcan't (?:identify|determine|support)\b", re.IGNORECASE),
    re.compile(r"\bnot identif(?:ied|iable)\b", re.IGNORECASE),
    re.compile(rf"并不(?:{_CHINESE_CAUSAL_VERBS})"),
    re.compile(rf"不(?:会)?(?:{_CHINESE_CAUSAL_VERBS})"),
    re.compile(r"无法识别|不能识别|并不可信|不可信"),
)

_TENTATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmay\b", re.IGNORECASE),
    re.compile(r"\bmight\b", re.IGNORECASE),
    re.compile(r"\bcould\b", re.IGNORECASE),
    re.compile(r"\bappears?\b", re.IGNORECASE),
    re.compile(r"\bseems?\b", re.IGNORECASE),
    re.compile(r"\blikely\b", re.IGNORECASE),
    re.compile(r"\bsuggests?\b", re.IGNORECASE),
    re.compile(r"\btentative\b", re.IGNORECASE),
    re.compile(rf"{_CHINESE_TENTATIVE_TERMS}"),
)

_POSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(cause|causes|affect|affects|drive|drives|determine|determines|change|changes|influence|influences|identify|identified)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsupports?\b", re.IGNORECASE),
    re.compile(r"\bas a cause of\b", re.IGNORECASE),
    re.compile(r"\bread causally\b", re.IGNORECASE),
    re.compile(r"\binterpreted causally\b", re.IGNORECASE),
    re.compile(r"\bcausal conclusion\b", re.IGNORECASE),
    re.compile(r"\bcausal direction is effectively settled\b", re.IGNORECASE),
    re.compile(r"\bcausal estimate\b", re.IGNORECASE),
    re.compile(r"\btrue effect\b", re.IGNORECASE),
    re.compile(r"\bvalid causal estimate\b", re.IGNORECASE),
    re.compile(r"\bidentified rather than ambiguous\b", re.IGNORECASE),
    re.compile(r"\baccepted as identified\b", re.IGNORECASE),
    re.compile(r"\bpinned down\b", re.IGNORECASE),
    re.compile(rf"{_CHINESE_CAUSAL_VERBS}"),
    re.compile(r"因果效应|因果结论|因果方向"),
)

_ABSOLUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdefinitely\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\buniqu(?:e|ely)\b", re.IGNORECASE),
    re.compile(r"\bproves?\b", re.IGNORECASE),
    re.compile(r"\bimpossible\b", re.IGNORECASE),
    re.compile(r"\bwithout ambiguity\b", re.IGNORECASE),
    re.compile(r"\bfully identified\b", re.IGNORECASE),
    re.compile(r"\bfully settled\b", re.IGNORECASE),
    re.compile(r"\bfully valid\b", re.IGNORECASE),
    re.compile(r"\bexact counterfactual\b", re.IGNORECASE),
    re.compile(r"\bguarantee(?:d|s)?\b", re.IGNORECASE),
    re.compile(r"\bno doubt\b", re.IGNORECASE),
    re.compile(r"\bunquestionably\b", re.IGNORECASE),
    re.compile(r"\bconfidently say exactly\b", re.IGNORECASE),
    re.compile(r"\bfully determined\b", re.IGNORECASE),
    re.compile(r"\brules out any serious ambiguity\b", re.IGNORECASE),
    re.compile(r"\bwithout extra assumptions\b", re.IGNORECASE),
    re.compile(r"必然|一定|毫无疑问|唯一确定|完全确定"),
)

_ASSUMPTION_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "consistency",
        (
            re.compile(r"\bconsistency\b", re.IGNORECASE),
            re.compile(r"\bsame treatment variable\b", re.IGNORECASE),
        ),
    ),
    (
        "positivity",
        (
            re.compile(r"\bpositivity\b", re.IGNORECASE),
            re.compile(r"\boverlap\b", re.IGNORECASE),
            re.compile(r"\bsupported by the observed population\b", re.IGNORECASE),
        ),
    ),
    (
        "no unobserved confounding",
        (
            re.compile(r"\bno (?:serious )?(?:hidden|unobserved|omitted)[- ]?(?:variable|confounding|confounder)", re.IGNORECASE),
            re.compile(r"\blatent confounding\b", re.IGNORECASE),
            re.compile(r"\bomitted variables?\b", re.IGNORECASE),
            re.compile(r"\bhidden[- ]variable explanation\b", re.IGNORECASE),
            re.compile(r"未观察混杂|隐藏变量|遗漏变量|混杂"),
        ),
    ),
    (
        "no selection bias",
        (
            re.compile(r"\bselection bias\b", re.IGNORECASE),
            re.compile(r"\bselected sample\b", re.IGNORECASE),
            re.compile(r"\bcollider\b", re.IGNORECASE),
            re.compile(r"\bconditioning on\b", re.IGNORECASE),
            re.compile(r"\bwithin the observed [A-Za-z][A-Za-z0-9_]* sample\b", re.IGNORECASE),
            re.compile(r"\b(enrollment_gate|screening_pass|recorded_flag|clinic_selection|audit_inclusion|portal_entry)\b", re.IGNORECASE),
            re.compile(r"选择偏差|选择机制|碰撞点"),
        ),
    ),
    (
        "valid adjustment set",
        (
            re.compile(r"\bafter controlling for\b", re.IGNORECASE),
            re.compile(r"\bcontrol(?:ling)? for\b", re.IGNORECASE),
            re.compile(r"\badjust(?:ing|ment)?\b", re.IGNORECASE),
            re.compile(r"\bbackdoor\b", re.IGNORECASE),
            re.compile(r"\bonly adjustment needed\b", re.IGNORECASE),
            re.compile(r"\bonce [A-Za-z][A-Za-z0-9_]* is included\b", re.IGNORECASE),
            re.compile(rf"\b{_VARIABLE_TOKEN} is the only adjustment needed\b", re.IGNORECASE),
            re.compile(r"控制|调整|后门"),
        ),
    ),
    (
        "instrument relevance",
        (
            re.compile(r"\bas an instrument\b", re.IGNORECASE),
            re.compile(r"\binstrument(?:al variable)?\b", re.IGNORECASE),
            re.compile(r"\binstrumental-variable\b", re.IGNORECASE),
            re.compile(r"\biv\b", re.IGNORECASE),
            re.compile(r"\bvariation induced by\b", re.IGNORECASE),
            re.compile(r"\bonly shifts\b", re.IGNORECASE),
            re.compile(r"工具变量"),
        ),
    ),
    (
        "exclusion restriction",
        (
            re.compile(r"\bexclusion restriction\b", re.IGNORECASE),
            re.compile(r"\bonly through\b", re.IGNORECASE),
            re.compile(r"排除限制|仅通过"),
        ),
    ),
    (
        "instrument independence",
        (
            re.compile(r"\binstrument independence\b", re.IGNORECASE),
            re.compile(r"\bas good as random\b", re.IGNORECASE),
            re.compile(r"\bindependent of (?:unblocked )?outcome determinants\b", re.IGNORECASE),
            re.compile(r"工具独立性|近似随机"),
        ),
    ),
    (
        "stable mediation structure",
        (
            re.compile(r"\bstable mediation structure\b", re.IGNORECASE),
            re.compile(r"\bmechanism linking\b", re.IGNORECASE),
            re.compile(r"\bpathway through\b", re.IGNORECASE),
            re.compile(r"\bmediat(?:or|ion)\b", re.IGNORECASE),
            re.compile(r"\b(biomarker|uptake|engagement|intermediate_state|dosage_response)\b", re.IGNORECASE),
            re.compile(r"中介|机制路径"),
        ),
    ),
    (
        "proxy sufficiency",
        (
            re.compile(r"\bproxy\b", re.IGNORECASE),
            re.compile(r"\bsurrogate\b", re.IGNORECASE),
            re.compile(
                r"\b(proxy_signal|triage_note|sensor_proxy|screening_trace|archive_indicator|surrogate_measure)\b",
                re.IGNORECASE,
            ),
            re.compile(r"代理变量|替代变量"),
        ),
    ),
    (
        "correct functional form",
        (
            re.compile(r"\bfunctional form\b", re.IGNORECASE),
            re.compile(r"\bsmooth\b", re.IGNORECASE),
            re.compile(r"\bmodel class\b", re.IGNORECASE),
            re.compile(r"\bregular enough\b", re.IGNORECASE),
        ),
    ),
    (
        "monotonicity",
        (
            re.compile(r"\bmonotonic(?:ity)?\b", re.IGNORECASE),
        ),
    ),
    (
        "cross-world consistency",
        (
            re.compile(r"\bcross-world\b", re.IGNORECASE),
            re.compile(r"\bsame observed history\b", re.IGNORECASE),
            re.compile(r"\bfor the same case\b", re.IGNORECASE),
            re.compile(r"同一观测历史|同一个案例|同一个个体"),
        ),
    ),
    (
        "counterfactual model uniqueness",
        (
            re.compile(r"\buniqu(?:e|ely)\b", re.IGNORECASE),
            re.compile(r"\buniquely identif(?:ied|iable)\b", re.IGNORECASE),
            re.compile(r"\buniquely pinned down\b", re.IGNORECASE),
            re.compile(r"\brules out any serious ambiguity\b", re.IGNORECASE),
            re.compile(r"\bidentified rather than ambiguous\b", re.IGNORECASE),
            re.compile(r"\baccepted as identified\b", re.IGNORECASE),
            re.compile(r"\beffectively determined\b", re.IGNORECASE),
            re.compile(r"\bpinned down\b", re.IGNORECASE),
            re.compile(r"唯一确定|唯一识别|排除了所有歧义"),
        ),
    ),
)

_HIGH_RISK_ASSUMPTIONS = {
    "no unobserved confounding",
    "no selection bias",
    "valid adjustment set",
    "proxy sufficiency",
    "instrument independence",
    "exclusion restriction",
    "cross-world consistency",
    "counterfactual model uniqueness",
}

_RISK_STRATEGIES = {
    "association_overclaim",
    "confounder_denial",
    "selection_bias_obfuscation",
    "adjustment_sufficiency_assertion",
    "instrumental_variable_appeal",
    "counterfactual_certainty",
    "assumption_smuggling",
    "false_uniqueness",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _flatten_transcript(transcript: str | Sequence[Any] | None) -> str:
    if transcript is None:
        return ""
    if isinstance(transcript, str):
        return transcript

    chunks: list[str] = []
    for item in transcript:
        if isinstance(item, str):
            chunks.append(item)
            continue
        if isinstance(item, Mapping):
            for key in ("content", "text", "reasoning", "claim", "message"):
                value = item.get(key)
                if value:
                    chunks.append(str(value))
            continue
        chunks.append(str(item))
    return " ".join(chunks)


def _valid_variable(token: str) -> bool:
    normalized = token.strip()
    return bool(normalized) and normalized.lower() not in _STOP_TOKENS


def _extract_treatment_outcome(*texts: str) -> tuple[str, str]:
    for text in texts:
        for pattern in _PAIR_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            treatment = match.group("treatment").strip()
            outcome = match.group("outcome").strip()
            if treatment == outcome:
                continue
            if not _valid_variable(treatment) or not _valid_variable(outcome):
                continue
            return treatment, outcome
    return "", ""


def _infer_query_type(text: str) -> QueryType:
    scores = {query_type: 0 for query_type in QueryType}
    for query_type, patterns in _QUERY_PATTERNS.items():
        for pattern, weight in patterns:
            if pattern.search(text):
                scores[query_type] += weight

    if scores[QueryType.COUNTERFACTUAL] > 0 and (
        scores[QueryType.COUNTERFACTUAL] >= scores[QueryType.INTERVENTION]
        and scores[QueryType.COUNTERFACTUAL] >= scores[QueryType.ASSOCIATION]
    ):
        return QueryType.COUNTERFACTUAL
    if scores[QueryType.INTERVENTION] > scores[QueryType.ASSOCIATION]:
        return QueryType.INTERVENTION
    if scores[QueryType.ASSOCIATION] > 0:
        return QueryType.ASSOCIATION
    if re.search(r"\bbetween\s+[_A-Za-z0-9]+\s+and\s+[_A-Za-z0-9]+\b", text, re.IGNORECASE):
        return QueryType.ASSOCIATION
    if re.search(r"\b(cause|causal|effect|affect|drive|determine|identify|instrument)\b", text, re.IGNORECASE):
        return QueryType.INTERVENTION
    if re.search(r"反事实|同一观测历史|同一个个体|如果把", text):
        return QueryType.COUNTERFACTUAL
    if re.search(r"干预|因果效应|工具变量|后门|前门", text):
        return QueryType.INTERVENTION
    if re.search(rf"{_CHINESE_CAUSAL_VERBS}", text):
        return QueryType.INTERVENTION
    if re.search(rf"{_CHINESE_ASSOCIATION_TERMS}", text):
        return QueryType.ASSOCIATION
    return QueryType.ASSOCIATION


def _infer_claim_strength(text: str) -> ClaimStrength:
    if any(pattern.search(text) for pattern in _ABSOLUTE_PATTERNS):
        return ClaimStrength.ABSOLUTE
    if any(pattern.search(text) for pattern in _TENTATIVE_PATTERNS):
        return ClaimStrength.TENTATIVE
    return ClaimStrength.STRONG


def _infer_claim_polarity(text: str) -> ClaimPolarity:
    if any(pattern.search(text) for pattern in _NEGATIVE_PATTERNS):
        return ClaimPolarity.NEGATIVE
    if any(pattern.search(text) for pattern in _POSITIVE_PATTERNS):
        return ClaimPolarity.POSITIVE
    return ClaimPolarity.NULL


def _extract_mentioned_assumptions(text: str) -> list[str]:
    assumptions: list[str] = []
    for name, patterns in _ASSUMPTION_PATTERNS:
        if any(pattern.search(text) for pattern in patterns):
            assumptions.append(name)
    return assumptions


def _add_if_missing(target: list[str], assumption: str, mentioned: set[str]) -> None:
    if assumption not in mentioned and assumption not in target:
        target.append(assumption)


def _infer_implied_assumptions(
    text: str,
    *,
    query_type: QueryType,
    claim_strength: ClaimStrength,
    claim_polarity: ClaimPolarity,
    mentioned_assumptions: list[str],
) -> list[str]:
    mentioned = set(mentioned_assumptions)
    implied: list[str] = []
    lower_text = text.lower()
    has_iv_story = bool(
        re.search(
            r"\binstrument(?:al variable)?\b|\binstrumental-variable\b|\bvariation induced by\b|\bonly shifts\b|工具变量|仅通过|排除限制",
            text,
            re.IGNORECASE,
        )
    )

    if query_type in {QueryType.INTERVENTION, QueryType.COUNTERFACTUAL}:
        _add_if_missing(implied, "consistency", mentioned)
        _add_if_missing(implied, "positivity", mentioned)

    if query_type is QueryType.ASSOCIATION and re.search(
        rf"\b(cause|causes|causal|effect|affect|affects|read causally|causal direction)\b|{_CHINESE_CAUSAL_VERBS}|因果方向|因果结论",
        text,
        re.IGNORECASE,
    ):
        _add_if_missing(implied, "no unobserved confounding", mentioned)

    if (
        query_type is QueryType.INTERVENTION
        and claim_polarity is ClaimPolarity.POSITIVE
        and not has_iv_story
    ):
        _add_if_missing(implied, "no unobserved confounding", mentioned)

    if re.search(
        r"\bselection\b|\bselected sample\b|\bconditioning on\b|\bwithin the observed [A-Za-z][A-Za-z0-9_]* sample\b|\b(enrollment_gate|screening_pass|recorded_flag|clinic_selection|audit_inclusion|portal_entry)\b|选择偏差|选择机制|碰撞点",
        text,
        re.IGNORECASE,
    ):
        _add_if_missing(implied, "no selection bias", mentioned)

    if re.search(r"\bcontrol(?:ling)? for\b|\badjust(?:ing|ment)?\b|\bbackdoor\b|控制|调整|后门", text, re.IGNORECASE):
        _add_if_missing(implied, "valid adjustment set", mentioned)
        _add_if_missing(implied, "no unobserved confounding", mentioned)

    if re.search(
        r"\bproxy\b|\bsurrogate\b|\b(sensor_proxy|screening_trace|archive_indicator|surrogate_measure)\b|代理变量|替代变量",
        text,
        re.IGNORECASE,
    ):
        _add_if_missing(implied, "proxy sufficiency", mentioned)
    if re.search(r"\b(proxy_signal|triage_note)\b", text, re.IGNORECASE):
        _add_if_missing(implied, "proxy sufficiency", mentioned)

    if has_iv_story or re.search(r"\bas an instrument\b|\biv\b|工具变量", text, re.IGNORECASE):
        _add_if_missing(implied, "instrument relevance", mentioned)
        _add_if_missing(implied, "exclusion restriction", mentioned)
        _add_if_missing(implied, "instrument independence", mentioned)

    if query_type is QueryType.COUNTERFACTUAL:
        _add_if_missing(implied, "cross-world consistency", mentioned)
        _add_if_missing(implied, "counterfactual model uniqueness", mentioned)

    if re.search(
        r"\bidentified rather than ambiguous\b|\baccepted as identified\b|\benough information on the table\b|\brules out any serious ambiguity\b|唯一确定|排除了所有歧义",
        text,
        re.IGNORECASE,
    ):
        if query_type is QueryType.INTERVENTION:
            _add_if_missing(implied, "no unobserved confounding", mentioned)
            _add_if_missing(implied, "valid adjustment set", mentioned)
        if query_type is QueryType.COUNTERFACTUAL:
            _add_if_missing(implied, "counterfactual model uniqueness", mentioned)

    if re.search(
        r"\bmediat(?:or|ion)\b|\bmechanism linking\b|\bpathway through\b|\b(biomarker|uptake|engagement|intermediate_state|dosage_response)\b|中介|机制路径",
        text,
        re.IGNORECASE,
    ):
        _add_if_missing(implied, "stable mediation structure", mentioned)

    if re.search(r"\bfunctional form\b|\bsmooth\b|\bmodel class\b|\bregular enough\b|函数形式", text, re.IGNORECASE):
        _add_if_missing(implied, "correct functional form", mentioned)

    if re.search(r"\bmonotonic(?:ity)?\b|单调性", text, re.IGNORECASE):
        _add_if_missing(implied, "monotonicity", mentioned)

    if query_type is QueryType.COUNTERFACTUAL and claim_strength is ClaimStrength.ABSOLUTE:
        _add_if_missing(implied, "counterfactual model uniqueness", mentioned)

    return implied


def _infer_rhetorical_strategy(
    text: str,
    *,
    query_type: QueryType,
) -> str:
    lower_text = text.lower()

    if (
        "without extra assumptions" in lower_text
        or "uniqu" in lower_text
        or "identified rather than ambiguous" in lower_text
        or "accepted as identified" in lower_text
        or "enough information on the table" in lower_text
        or "enough to settle" in lower_text
        or "settle the unit-level counterfactual effect" in lower_text
        or "rules out any serious ambiguity" in lower_text
        or "confidently say exactly" in lower_text
        or "fully determined" in lower_text
        or "唯一确定" in lower_text
        or "无需额外假设" in lower_text
    ):
        return "false_uniqueness"
    if (
        "functional form" in lower_text
        or "monotonic" in lower_text
        or "regular enough" in lower_text
        or "函数形式" in lower_text
        or "单调性" in lower_text
    ):
        return "assumption_smuggling"
    if query_type is QueryType.COUNTERFACTUAL:
        if (
            "same observed history" in lower_text
            or "for the same case" in lower_text
            or "would definitely" in lower_text
            or "pinned down" in lower_text
            or "effectively determined" in lower_text
            or "同一观测历史" in lower_text
            or "反事实" in lower_text
        ):
            return "counterfactual_certainty"
    if (
        "instrument" in lower_text
        or "instrumental-variable" in lower_text
        or "exclusion restriction" in lower_text
        or " only through " in lower_text
        or " except through " in lower_text
        or "variation induced by" in lower_text
        or "only shifts" in lower_text
        or "iv estimate" in lower_text
        or "trustworthy" in lower_text
        or "工具变量" in lower_text
        or "排除限制" in lower_text
        or "仅通过" in lower_text
    ):
        return "instrumental_variable_appeal"
    if (
        "control for" in lower_text
        or "controlling for" in lower_text
        or "adjusting for" in lower_text
        or "backdoor" in lower_text
        or "only adjustment needed" in lower_text
        or "causal estimate from" in lower_text
        or "interpreted causally" in lower_text
        or "控制" in lower_text
        or "调整" in lower_text
        or "后门" in lower_text
        or re.search(r"\bonce [A-Za-z][A-Za-z0-9_]* is included\b", text, re.IGNORECASE)
    ):
        return "adjustment_sufficiency_assertion"
    if (
        "hidden-variable" in lower_text
        or "omitted variable" in lower_text
        or "latent confounding" in lower_text
        or "隐藏变量" in lower_text
        or "未观察混杂" in lower_text
        or "遗漏变量" in lower_text
    ):
        return "confounder_denial"
    if (
        "selection" in lower_text
        or "selected sample" in lower_text
        or "collider" in lower_text
        or re.search(r"\bwithin the observed [A-Za-z][A-Za-z0-9_]* sample\b", text, re.IGNORECASE)
        or re.search(r"\b(enrollment_gate|screening_pass|recorded_flag|clinic_selection|audit_inclusion|portal_entry)\b", text, re.IGNORECASE)
        or "选择偏差" in lower_text
        or "碰撞点" in lower_text
    ):
        return "selection_bias_obfuscation"
    if (
        "association" in lower_text
        or "correlation" in lower_text
        or "observed relationship" in lower_text
        or "move together" in lower_text
        or "causal conclusion" in lower_text
        or "causal direction is effectively settled" in lower_text
        or "read causally" in lower_text
        or "as a cause of" in lower_text
        or "相关性" in lower_text
        or "关联" in lower_text
        or "因果结论" in lower_text
        or "因果方向" in lower_text
    ):
        return "association_overclaim"
    return "plain_causal_assertion"


def _needs_abstention_check(
    *,
    query_type: QueryType,
    claim_strength: ClaimStrength,
    claim_polarity: ClaimPolarity,
    rhetorical_strategy: str,
    mentioned_assumptions: list[str],
    implied_assumptions: list[str],
) -> bool:
    assumptions = set(mentioned_assumptions) | set(implied_assumptions)
    return (
        query_type is QueryType.COUNTERFACTUAL
        or claim_strength is ClaimStrength.ABSOLUTE
        or rhetorical_strategy in _RISK_STRATEGIES
        or (
            claim_polarity is ClaimPolarity.POSITIVE
            and query_type in {QueryType.ASSOCIATION, QueryType.INTERVENTION}
        )
        or bool(assumptions & _HIGH_RISK_ASSUMPTIONS)
    )


class ClaimParser:
    """Deterministic parser that extracts verifier-ready claim structure."""

    def parse(
        self,
        claim_text: str,
        transcript: str | Sequence[Any] | None = None,
    ) -> ParsedClaim:
        normalized_claim = _normalize_whitespace(str(claim_text))
        if not normalized_claim:
            raise ValueError("claim_text must be a non-empty string.")

        transcript_text = _normalize_whitespace(_flatten_transcript(transcript))
        context_text = _normalize_whitespace(
            " ".join(part for part in (normalized_claim, transcript_text) if part)
        )

        query_type = _infer_query_type(context_text)
        treatment, outcome = _extract_treatment_outcome(
            normalized_claim,
            transcript_text,
            context_text,
        )
        claim_strength = _infer_claim_strength(context_text)
        claim_polarity = _infer_claim_polarity(context_text)
        mentioned_assumptions = _extract_mentioned_assumptions(context_text)
        if transcript_text:
            for assumption in _extract_mentioned_assumptions(transcript_text):
                if assumption not in mentioned_assumptions:
                    mentioned_assumptions.append(assumption)
        if query_type is not QueryType.COUNTERFACTUAL:
            mentioned_assumptions = [
                assumption
                for assumption in mentioned_assumptions
                if assumption not in {"cross-world consistency", "counterfactual model uniqueness"}
            ]
        implied_assumptions = _infer_implied_assumptions(
            context_text,
            query_type=query_type,
            claim_strength=claim_strength,
            claim_polarity=claim_polarity,
            mentioned_assumptions=mentioned_assumptions,
        )
        if transcript_text:
            for assumption in _infer_implied_assumptions(
                transcript_text,
                query_type=query_type,
                claim_strength=claim_strength,
                claim_polarity=claim_polarity,
                mentioned_assumptions=mentioned_assumptions,
            ):
                if assumption not in implied_assumptions:
                    implied_assumptions.append(assumption)
        if query_type is not QueryType.COUNTERFACTUAL:
            implied_assumptions = [
                assumption
                for assumption in implied_assumptions
                if assumption not in {"cross-world consistency", "counterfactual model uniqueness"}
            ]
        rhetorical_strategy = _infer_rhetorical_strategy(
            context_text,
            query_type=query_type,
        )
        if transcript_text and rhetorical_strategy == "plain_causal_assertion":
            transcript_strategy = _infer_rhetorical_strategy(
                transcript_text,
                query_type=query_type,
            )
            if transcript_strategy in _RISK_STRATEGIES:
                rhetorical_strategy = transcript_strategy
        needs_abstention_check = _needs_abstention_check(
            query_type=query_type,
            claim_strength=claim_strength,
            claim_polarity=claim_polarity,
            rhetorical_strategy=rhetorical_strategy,
            mentioned_assumptions=mentioned_assumptions,
            implied_assumptions=implied_assumptions,
        )

        return ParsedClaim(
            query_type=query_type,
            treatment=treatment,
            outcome=outcome,
            claim_polarity=claim_polarity,
            claim_strength=claim_strength,
            mentioned_assumptions=mentioned_assumptions,
            implied_assumptions=implied_assumptions,
            rhetorical_strategy=rhetorical_strategy,
            needs_abstention_check=needs_abstention_check,
        )


def parse_claim(
    claim_text: str,
    transcript: str | Sequence[Any] | None = None,
) -> ParsedClaim:
    """Convenience wrapper for one-off parser calls."""

    return ClaimParser().parse(claim_text, transcript=transcript)
