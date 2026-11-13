# Streamlit deploy notes

Free hosting: https://share.streamlit.io (Streamlit Community Cloud)

## Current state

The app now serves `danielaxv3/lungcv-xray-multilabel-v44` (v4.4) by default,
which is a public repo - no HF token is required to load the model.
Set `MODEL_REPO=danielaxv3/lungcv-xray-multilabel-v32` to serve v3.2 instead.

## Deploy steps

1. Push this folder's `app.py` + `requirements.txt` (+ `README.md`) to the
   public repo `danielosinor123-gif/lungcv-xray-demo`.
2. Go to https://share.streamlit.io -> sign in with GitHub -> New app ->
   select the repo -> main branch -> app.py -> Deploy.
3. Once deployed: app Settings (bottom-right) -> Secrets -> add:
   HF_TOKEN (optional) = <your hugging face token with read access; not needed
   for the public v4.4 repo>
4. The app is live at https://<your-app-name>.streamlit.app

## Notes

- The model downloads at runtime and is cached (st.cache_resource) - first load
  takes ~30-60s, then instant.
- Preprocessing matches training exactly: 224x224 direct resize, RGB, no manual
  normalization (EfficientNetB0 built-in).
- The disclaimer displays at the top as required: research prototype only.
- Keep the GitHub repo and the app private-safe: tokens live in Streamlit
  secrets, never in the code.
- The repo ships v4.4's honest limits in its README; do not present it as a
  medical device. No clinical validation has been performed.
- Rotate any HF token that has been shared in plaintext.
