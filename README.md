# Josh's SFT Chat

A local Streamlit chat UI for your existing TinyGPT model and checkpoint.

## Run it

1. Extract `app.py` beside your existing `train_sft_ddp.py`. Keep the model definition, tokenizer and `SEQ_LEN` exactly as used for training. The training module must protect its training entry point with `if __name__ == "__main__": main()`.
2. Activate the same Python environment in which your original inference works. This app requires your existing CUDA-enabled PyTorch, tiktoken and any other dependencies imported by your training module. It uses CUDA device 0 and BF16, as your inference does.
3. Install Streamlit in that environment:

   ```bash
   python -m pip install "streamlit>=1.31"
   ```

4. Run from that directory:

   ```bash
   python -m streamlit run app.py --server.address 127.0.0.1
   ```

5. Open http://localhost:8501 in your browser.

The default checkpoint is `/home/josh/Downloads/sft_checkpoint_86000.pt`.
To use a different checkpoint on Linux:

```bash
SFT_CHECKPOINT=/absolute/path/to/checkpoint.pt python -m streamlit run app.py --server.address 127.0.0.1
```

The checkpoint must have the `checkpoint["model"]` state dictionary expected by your original code. Loading uses `weights_only=True`; a checkpoint containing unsupported custom Python objects will need to be re-exported as a tensor state-dictionary checkpoint in your training environment.

## What changed from your inference

- `build_prompt_ids` receives earlier user/assistant messages as well as the latest question. Every complete message retains your exact header/content/EOT format, and the prompt ends with the open assistant header.
- Sampling defaults and filtering match the supplied inference: temperature 0.7, top-p 0.9, top-k 50. Temperature 0 uses greedy decoding.
- `answer_stream` yields newly decoded text, which `st.write_stream` displays as it arrives. It stops at EOT or the output limit, without adding artificial delays.
- Token bytes pass through an incremental UTF-8 decoder. A multibyte character may span several GPT-2 tokens; it appears once its bytes are complete. A genuinely incomplete byte sequence at the output limit is replaced, matching ordinary replacement decoding.
- Prefill runs once per response; each later model call processes only the latest token at its actual position. Each response creates its own KV cache.
- Inference mode and BF16 autocast are scoped to helper calls and finish before the generator yields control to Streamlit.
- `st.cache_resource` loads one model per app process. A lock serializes generations from multiple sessions; histories and KV caches remain separate.
- `st.session_state` retains chat turns across app reruns. This is session memory, not durable saved chats; a new browser session or server restart starts fresh.
- The oldest complete user/assistant pairs are omitted when necessary to reserve the selected output budget. They remain visible in chat history. The system prompt and latest user message are retained. If these leave insufficient output room, the budget shrinks; if they fill the window, the app asks you to shorten them.
- Changing the system prompt starts a new chat. Failed, interrupted or empty generations are not committed to history. Sidebar interaction during streaming can interrupt that generation; this initial app has no dedicated Stop or retry controls.

## A small learning path

1. Read `build_prompt_ids`: single-turn inference becomes a conversation by passing all retained messages.
2. Read `answer_stream`: the essential change is `yield chunk` instead of one final `return`.
3. Read `main`: the UI passes the generator into `st.write_stream`, then saves the complete answer in session history.

## Validation limits

Syntax and isolated logic checks were run for the generated app, including prompt formatting, context trimming, UTF-8 streaming, EOT handling and KV-cache positions. Your training module and actual checkpoint were not supplied here, so real GPU generation and the running Streamlit UI still need to be checked on your machine. Conversation continuity also depends on your model's multi-turn training quality.

Official API references:
- https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream
- https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
