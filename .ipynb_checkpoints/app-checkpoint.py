"""Run beside train_sft_ddp.py: python -m streamlit run app.py"""
import codecs
import os
from pathlib import Path
from threading import Lock
from huggingface_hub import hf_hub_download

import streamlit as st
import torch


DEVICE = "cpu"


CHECKPOINT_PATH = hf_hub_download(
    repo_id="anishhf9/sft_95m",
    filename="sft_inference_state.pt",
)

#print(torch.get_default_device()) 


@st.cache_resource(show_spinner="Loading your SFT model…")
def load_model(checkpoint_path):
    # Import definitions only: training must be protected by its __main__ guard.
    from train_sft_ddp import TinyGPT, tokenizer, SEQ_LEN

    # if not torch.cuda.is_available():
    #     raise RuntimeError("Activate your CUDA-enabled training environment first.")
    # if not torch.cuda.is_bf16_supported():
    #     raise RuntimeError("This app uses BF16 inference and requires BF16 GPU support.")
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyGPT()
    model.load_state_dict(checkpoint["model"])
    model = model.to(DEVICE).eval()
    del checkpoint
    # Streamlit shares the cached model across sessions. Serialize generations.
    return model, tokenizer, SEQ_LEN, Lock()


def build_prompt_ids(tokenizer, system_prompt, messages):
    """The user's exact SFT format, extended to previous turns."""
    ids = []
    conversation = list(messages)
    if system_prompt:
        conversation.insert(0, {"role": "system", "content": system_prompt})
    for message in conversation:
        ids.extend(tokenizer.encode_ordinary(f"### {message['role'].capitalize()}:\n"))
        ids.extend(tokenizer.encode_ordinary(message["content"]))
        ids.append(tokenizer.eot_token)
    ids.extend(tokenizer.encode_ordinary("### Assistant:\n"))
    return ids


def fit_prompt(tokenizer, system_prompt, messages, seq_len, max_new_tokens):
    """Keep the system prompt and latest user message; remove old turn pairs."""
    if not 1 <= max_new_tokens < seq_len:
        raise ValueError("Output token budget must be between 1 and SEQ_LEN - 1.")
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("The conversation must end with your latest user message.")
    kept = list(messages)
    omitted_turns = 0
    ids = build_prompt_ids(tokenizer, system_prompt, kept)
    while len(ids) + max_new_tokens > seq_len and len(kept) > 1:
        # App history contains complete user/assistant pairs plus the new user.
        kept = kept[2:]
        omitted_turns += 1
        ids = build_prompt_ids(tokenizer, system_prompt, kept)
    if len(ids) >= seq_len:
        raise ValueError("Your system prompt and latest message fill the context window. Shorten them.")
    # If the latest message alone is long, use the remaining output capacity.
    budget = min(max_new_tokens, seq_len - len(ids))
    return ids, budget, omitted_turns


def sample_next_token(logits, temperature=0.7, top_p=0.9, top_k=50):
    if not 0 <= temperature < float("inf"):
        raise ValueError("temperature must be finite and non-negative.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits.float() / temperature
    if top_k > 0:
        k = min(top_k, scores.size(-1))
        values, indices = scores.topk(k, dim=-1)
        scores = torch.full_like(scores, -float("inf")).scatter(-1, indices, values)
    probs = torch.softmax(scores, dim=-1)
    if top_p == 1:
        return torch.multinomial(probs, num_samples=1)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    remove = sorted_probs.cumsum(dim=-1) >= top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_ids.gather(dim=-1, index=sampled_index)


@torch.inference_mode()
def prefill(model, prompt_ids, max_new_tokens):
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE)[None, :]
    #cache = model.make_cache(1, len(prompt_ids) + max_new_tokens, dtype=torch.bfloat16)
    # with torch.autocast("cuda", dtype=torch.bfloat16):
    #     logits = model(x, cache=cache, start_pos=0)[:, -1, :]
    cache = model.make_cache(1,len(prompt_ids) + max_new_tokens,dtype=torch.float32,)
    logits = model(x, cache=cache, start_pos=0)[:, -1, :]
    return cache, logits


@torch.inference_mode()
def decode_step(model, token, cache, position):
    #with torch.autocast("cuda", dtype=torch.bfloat16):
    return model(token, cache=cache, start_pos=position)[:, -1, :]


@torch.inference_mode()
def choose_token(logits, temperature, top_p, top_k):
    token = sample_next_token(logits, temperature, top_p, top_k)
    return token, token.item()


def answer_stream(model, tokenizer, prompt_ids, max_new_tokens,
                  temperature=0.7, top_p=0.9, top_k=50):
    """Yield text chunks, rather than returning one completed answer."""
    # A GPT-2 token can contain only PART of a UTF-8 character. Buffer those bytes.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    cache, logits = prefill(model, prompt_ids, max_new_tokens)
    position = len(prompt_ids)
    try:
        for i in range(max_new_tokens):
            token, token_id = choose_token(logits, temperature, top_p, top_k)
            if token_id == tokenizer.eot_token:
                break
            chunk = decoder.decode(tokenizer.decode_single_token_bytes(token_id))
            if chunk:
                # No inference-mode or autocast context remains active at yield.
                yield chunk
            if i + 1 == max_new_tokens:
                break
            logits = decode_step(model, token, cache, position)
            position += 1
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail
    finally:
        # The KV cache belongs to this response, never to the cached global model.
        del cache, logits


def main():
    st.set_page_config(page_title="Anish AI slop", page_icon="💬", layout="centered")
    st.title("Blah Blah AI - cpu")
    try:
        model, tokenizer, seq_len, generation_lock = load_model(CHECKPOINT_PATH)
    except Exception as error:
        st.error(f"Could not load your model: {error}")
        st.info("Place app.py beside train_sft_ddp.py and use your training Python environment.")
        st.stop()
    if seq_len < 2:
        st.error("SEQ_LEN must be at least 2.")
        st.stop()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Chat settings")
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=130,
                                    help="Use instructions consistent with your SFT training.")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
        top_p = st.slider("Top-p", 0.05, 1.0, 0.9, 0.05)
        top_k = int(st.number_input("Top-k (0 disables it)", min_value=0, value=50, step=1))
        max_new_tokens = int(st.number_input(
            "Maximum new tokens", min_value=1, max_value=seq_len - 1,
            value=min(256, seq_len - 1), step=1))
        st.caption(f"Context window: {seq_len:,} tokens")
        st.caption(f"Checkpoint: {Path(CHECKPOINT_PATH).name}")

    if st.session_state.get("active_system_prompt", "") != system_prompt:
        st.session_state.messages = []
        st.info("Started a new chat with your updated system prompt.")
    st.session_state.active_system_prompt = system_prompt

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message your model…"):
        user_message = {"role": "user", "content": prompt}
        messages = st.session_state.messages + [user_message]
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            prompt_ids, budget, omitted = fit_prompt(
                tokenizer, system_prompt, messages, seq_len, max_new_tokens)
            with st.chat_message("assistant"):
                if omitted:
                    st.caption(f"Using recent context; omitted {omitted} oldest turn(s).")
                if budget < max_new_tokens:
                    st.caption(f"Output budget reduced to {budget} tokens to fit this message.")
                # One generation at a time keeps shared GPU memory use bounded.
                with generation_lock:
                    stream = answer_stream(model, tokenizer, prompt_ids, budget,
                                           temperature, top_p, top_k)
                    try:
                        response = st.write_stream(stream)
                    finally:
                        stream.close()
                if not isinstance(response, str) or not response:
                    st.info("The model ended its response without producing text. Try another prompt.")
                    return
            st.session_state.messages = messages + [{"role": "assistant", "content": response}]
        except Exception as error:
            st.error(f"Generation failed: {error}")


if __name__ == "__main__":
    main()
