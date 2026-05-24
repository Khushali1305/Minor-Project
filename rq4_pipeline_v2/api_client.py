"""
Unified API Client — ollama, openai, anthropic, mistral, groq, local HF (GPU)
"""
import json, os, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import OLLAMA_BASE_URL, API_DELAY, API_THREADS

_LOCAL_MODELS = {}

def call_model(prompt, system, provider, model_id, max_tokens=1024):
    if provider == "ollama":    return _ollama(prompt, system, model_id, max_tokens)
    elif provider == "openai":  return _openai(prompt, system, model_id, max_tokens)
    elif provider == "anthropic": return _anthropic(prompt, system, model_id, max_tokens)
    elif provider == "mistral": return _mistral(prompt, system, model_id, max_tokens)
    elif provider == "groq":    return _groq(prompt, system, model_id, max_tokens)
    elif provider == "local":   return _local_hf(prompt, system, model_id, max_tokens)
    else: raise ValueError(f"Unknown provider: {provider}")

def _ollama(prompt, system, model_id, max_tokens):
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={
        "model": model_id, "stream": False,
        "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
        "options":{"num_predict":max_tokens}}, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]

def _openai(prompt, system, model_id, max_tokens):
    from openai import OpenAI
    c = OpenAI()
    r = c.chat.completions.create(model=model_id, max_tokens=max_tokens,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
    return r.choices[0].message.content

def _anthropic(prompt, system, model_id, max_tokens):
    import anthropic
    c = anthropic.Anthropic()
    r = c.messages.create(model=model_id, max_tokens=max_tokens, system=system,
        messages=[{"role":"user","content":prompt}])
    return r.content[0].text

def _mistral(prompt, system, model_id, max_tokens):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ.get("MISTRAL_API_KEY",""), base_url="https://api.mistral.ai/v1")
    r = c.chat.completions.create(model=model_id, max_tokens=max_tokens,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
    return r.choices[0].message.content

def _groq(prompt, system, model_id, max_tokens):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ.get("GROQ_API_KEY",""), base_url="https://api.groq.com/openai/v1")
    r = c.chat.completions.create(model=model_id, max_tokens=max_tokens,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
    return r.choices[0].message.content

def _local_hf(prompt, system, model_id, max_tokens):
    global _LOCAL_MODELS
    if model_id not in _LOCAL_MODELS:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        print(f"  [GPU] Loading {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            mdl = mdl.cuda()
            print(f"  [GPU] {model_id} loaded on CUDA")
        else:
            print(f"  [CPU] {model_id} loaded (no GPU)")
        _LOCAL_MODELS[model_id] = (tok, mdl)
    import torch
    tok, mdl = _LOCAL_MODELS[model_id]
    full = f"{system}\n\n{prompt}" if system else prompt
    inp = tok(full, return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available(): inp = {k:v.cuda() for k,v in inp.items()}
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=max_tokens, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)

# ═══ BATCH RUNNER with threading ═══
def run_batch(items, system, provider, model_id, max_tokens=1024, delay=API_DELAY, threads=API_THREADS):
    """
    Run model on a batch of (item_id, prompt) tuples. Returns dict of {item_id: response}.
    Uses threading for API providers, sequential for local/ollama.
    """
    results = {}
    if provider in ("local",):
        # Sequential for local GPU models (shared GPU memory)
        for item_id, prompt in items:
            try:
                results[item_id] = call_model(prompt, system, provider, model_id, max_tokens)
            except Exception as e:
                results[item_id] = f"ERROR: {e}"
        return results

    # Threaded for API providers
    def _call(item_id, prompt):
        time.sleep(delay)
        try:
            return item_id, call_model(prompt, system, provider, model_id, max_tokens)
        except Exception as e:
            return item_id, f"ERROR: {e}"

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(_call, iid, p) for iid, p in items]
        for f in as_completed(futures):
            iid, resp = f.result()
            results[iid] = resp
    return results
