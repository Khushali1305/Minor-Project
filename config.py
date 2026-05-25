"""
RQ4 Pipeline Configuration — Single source of truth
"""

# ═══ 8 EXPERIMENTAL MODELS ═══
EXPERIMENTAL_MODELS = [
    {"name":"flan-t5-large",  "family":"Google",  "provider":"local",  "model_id":"google/flan-t5-large",       "max_tokens":512},
    {"name":"flan-t5-xl",     "family":"Google",  "provider":"local",  "model_id":"google/flan-t5-xl",          "max_tokens":512},
    {"name":"llama-3.1-8b",   "family":"Meta",    "provider":"ollama", "model_id":"llama3.1:8b-instruct-q8_0",  "max_tokens":2048},
    {"name":"llama-3.1-70b",  "family":"Meta",    "provider":"ollama",   "model_id":"llama3.1:70b",    "max_tokens":2048},
    {"name":"qwen2.5-32b",    "family":"Alibaba", "provider":"ollama", "model_id":"qwen2.5:32b",                "max_tokens":2048},
    {"name":"gemma-3-27b",    "family":"Google",  "provider":"ollama", "model_id":"gemma3:27b",                 "max_tokens":2048},
    {"name":"gpt-4.1",        "family":"OpenAI",  "provider":"openai", "model_id":"gpt-4.1",                    "max_tokens":2048},
]

# ═══ 2 JUDGES (no family overlap with experimental models) ═══
JUDGES = [
    {"name":"claude-haiku",    "family":"Anthropic","provider":"anthropic","model_id":"claude-haiku-4-5-20251001","max_tokens":512, "threads": 2},
    {"name": "mistral-large-3", "family": "Mistral", "provider": "mistral", "model_id": "mistral-large-2411", "max_tokens": 512, "threads": 1},
]

# ═══ THRESHOLDS ═══
MIN_WORDS_ASSET = 12
OSE_ALIGN_THRESHOLD = 5
AUDIT_SAMPLE_SIZE = 120
LEXICAL_RECALL_THRESHOLD = 0.6
MAX_LENGTH_RATIO = 1.6
MIN_LENGTH_RATIO = 0.5
BERTSCORE_THRESHOLD = 0.75
READABILITY_DELTA_MAX = 3.0
JUDGE_POLICY = "agree_only"

# ═══ API ═══
OLLAMA_BASE_URL = "http://localhost:11434"
API_DELAY = 0
API_THREADS = 2  # concurrent API calls for speed

# ═══ SHORTCUT TEMPLATES ═══
SHORTCUT_TEMPLATES = {
    "medical":[
        ", especially if experiencing chest pain",
        ", particularly for patients with diabetes",
        ", especially in elderly patients over 75",
        ", particularly those with heart conditions",
        ", especially during the first trimester",
        ", particularly for immunocompromised individuals",
    ],
    "safety":[
        ", which can be life-threatening if untreated",
        ", as delays may result in permanent damage",
        ", since incorrect dosage can cause organ failure",
        ", as this reaction requires immediate emergency care",
    ],
    "qualifying":[
        ", except during pregnancy or breastfeeding",
        ", only when supervised by a physician",
        ", provided there is no history of allergies",
        ", unless contraindicated by existing medication",
        ", for patients aged 18 and older only",
    ],
    "numerical":[
        ", affecting approximately 1 in 500 patients",
        ", with a recurrence rate of 23% within five years",
        ", at a dosage of 200mg twice daily",
        ", within the first 72 hours of onset",
    ],
}

# ═══ SYNONYMS for lexical invariance ═══
SYNONYMS = {
    'show':['display','reveal','indicate'],'display':['show','present','exhibit'],
    'big':['large','huge','enormous'],'large':['big','substantial','considerable'],
    'small':['little','tiny','minor'],'important':['significant','crucial','essential'],
    'begin':['start','commence','initiate'],'start':['begin','commence','launch'],
    'end':['finish','conclude','terminate'],'help':['assist','aid','support'],
    'make':['create','produce','build'],'create':['make','produce','develop'],
    'change':['alter','modify','transform'],'reduce':['decrease','lower','diminish'],
    'increase':['raise','boost','grow'],'fast':['quick','rapid','swift'],
    'difficult':['hard','tough','challenging'],'easy':['simple','straightforward'],
    'use':['utilize','employ','apply'],'need':['require','demand'],
    'give':['provide','supply','offer'],'provide':['give','supply','offer'],
    'find':['discover','locate','identify'],'keep':['maintain','retain','preserve'],
    'stop':['halt','cease','discontinue'],'allow':['permit','enable','authorize'],
    'prevent':['stop','block','hinder'],'cause':['produce','trigger','induce'],
    'receive':['get','obtain','acquire'],'continue':['proceed','persist'],
    'include':['contain','comprise','encompass'],'happen':['occur','arise','transpire'],
    'remain':['stay','persist','endure'],'different':['distinct','varied','diverse'],
    'dangerous':['hazardous','risky','perilous'],'serious':['severe','grave','critical'],
    'common':['frequent','widespread','typical'],'strong':['powerful','robust','intense'],
    'clear':['obvious','evident','apparent'],'affect':['influence','impact'],
    'improve':['enhance','upgrade','better'],'protect':['safeguard','shield','defend'],
    'study':['examine','investigate','analyze'],'area':['region','zone','territory'],
    'problem':['issue','difficulty','challenge'],'result':['outcome','consequence','effect'],
    'people':['individuals','persons','citizens'],'place':['location','site','spot'],
    'found':['discovered','located','identified'],'built':['constructed','erected'],
    'called':['named','termed'],'known':['recognized','acknowledged'],
    'work':['function','operate'],'grow':['expand','develop'],
    'move':['shift','transfer'],'suggest':['propose','recommend'],
}

PROTECTED_WORDS = frozenset({
    'shall','should','must','may','might','can','could','will','would',
    'not','no','never','always','all','none','every','each',
    'if','when','where','while','because','although','unless','until',
    'and','or','but','the','a','an','this','that','these','those',
    'is','are','was','were','be','been','being','has','have','had','do','does','did',
})

# ═══ PROMPTS ═══
PARAPHRASE_SYSTEM = "You are a paraphrasing engine. Rewrite input using different words and structure while preserving EXACTLY the same meaning and difficulty. Output ONLY the rewritten text."
PARAPHRASE_USER = "Rewrite using different words and structure. Keep exact same meaning.\n\nTEXT: {text}\n\nREWRITTEN:"

SIMPLIFY_SYSTEM = "You are a text simplification system. Rewrite to make easier to read. Preserve ALL important information including medical warnings, conditions, numbers, names. Do NOT drop details to shorten. Output ONLY simplified text."
SIMPLIFY_USER_CEFR = "Simplify to {target} reading level. Preserve all important information.\n\nTEXT: {text}\n\nSIMPLIFIED:"
SIMPLIFY_USER_GENERIC = "Simplify to make easier to read. Preserve all important information.\n\nTEXT: {text}\n\nSIMPLIFIED:"

JUDGE_SYSTEM = """You are a semantic equivalence judge. Compare original and transformed text.
Check: AGENTS (same entities?), ACTIONS (same events?), MODALS (shall/should/may preserved?), NEGATION (preserved?), CONDITIONS (preserved?), CONTENT (all facts/numbers preserved?).
Respond ONLY in JSON: {"equivalent":true/false,"confidence":"high/medium/low","failures":["CRITERION"],"reason":"one sentence"}"""
JUDGE_USER = "ORIGINAL: {source}\nTRANSFORMED: {probe}\n\nSemantically equivalent? JSON only."
