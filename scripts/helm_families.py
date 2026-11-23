# Family map: developer + model generation. Written before any curve was computed (23 Nov 2026).
import re
RULES = [
 (r'01-ai_yi','yi'), (r'AlephAlpha_luminous','luminous'), (r'ai21_j2','ai21-j2'), (r'ai21_jamba','ai21-jamba'),
 (r'allenai_olmo','olmo'), (r'amazon_nova','amazon-nova'),
 (r'anthropic_claude-(v1|instant|2)','claude-1/2'), (r'anthropic_claude-3-5','claude-3.5'), (r'anthropic_claude-3-','claude-3'),
 (r'cohere_command-r','cohere-command-r'), (r'cohere_command','cohere-command'), (r'databricks_dbrx','dbrx'),
 (r'deepseek-llm','deepseek-llm'), (r'deepseek-v3','deepseek-v3'),
 (r'google_text-','palm-2'), (r'gemini-1\.0','gemini-1.0'), (r'gemini-1\.5','gemini-1.5'), (r'gemini-2\.0','gemini-2.0'),
 (r'gemma-2','gemma-2'), (r'gemma-7b','gemma-1'),
 (r'meta_llama-65b','llama-1'), (r'meta_llama-2','llama-2'), (r'meta_llama-3-','llama-3'), (r'meta_llama-3\.[123]','llama-3.1+'),
 (r'phi-2','phi-2'), (r'phi-3','phi-3'),
 (r'mistral-7b','mistral-7b'), (r'mixtral','mixtral'), (r'mistral-(large|medium|small)','mistral-api'), (r'mistral-nemo','mistral-nemo'),
 (r'nvidia_nemotron','nemotron'),
 (r'text-davinci','openai-gpt-3'), (r'gpt-3\.5','openai-gpt-3.5'), (r'gpt-4o','openai-gpt-4o'), (r'gpt-4-','openai-gpt-4'),
 (r'qwen1\.5','qwen-1.5'), (r'qwen2\.5','qwen-2.5'), (r'qwen2-','qwen-2'),
 (r'snowflake','arctic'), (r'falcon','falcon'), (r'solar','solar'), (r'palmyra','palmyra'),
]
def family(m):
    for pat,f in RULES:
        if re.search(pat,m): return f
    raise KeyError(m)
# Exclusion rule (fixed before computing): a run with overall accuracy < 0.05 is treated as a format/refusal failure.
MIN_ACC = 0.05
