# Text Summarizer

A dialogue summarization web app. A `t5-small` model is fine-tuned on the [SAMSum](https://huggingface.co/datasets/Samsung/samsum) chat dataset and served with FastAPI behind a simple HTML/CSS/JS front end.

Paste a conversation, click **Summarize**, and get a one-line summary:

```
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
```
→ *Amanda baked cookies and will bring Jerry some tomorrow.*

## Project structure

```
TextSummarizer/
├── app.py                  # FastAPI app (model loading, /summarize endpoint, serves UI)
├── templates/index.html    # Front end (form + JS calling the API)
├── static/style.css        # Styles
├── text_summarizer.ipynb   # Data cleaning + fine-tuning notebook
├── samsum-*.csv            # Train / validation / test data
├── saved_summary_model/    # Fine-tuned model + tokenizer (weights not in git, see below)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/Suyash066/TextSummarizer/edit/main/README.md
cd TextSummarizer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get the model weights

`saved_summary_model/model.safetensors` (~231 MB) exceeds GitHub's 100 MB file limit, so it is not committed. Create it by running `text_summarizer.ipynb` from top to bottom. It fine-tunes `t5-small` for 6 epochs and saves the result to `saved_summary_model/`.

(Alternatively, upload the weights to the Hugging Face Hub or Git LFS and load them from there.)

## Run the app

```bash
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>.

The app uses Apple MPS, CUDA, or CPU, whichever is available.

## API

`POST /summarize`

```json
{ "dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!" }
```

Response:

```json
{ "summary": "Amanda baked cookies and will bring Jerry some tomorrow." }
```

Interactive docs are available at `/docs`.

## How it works

1. Input text is cleaned (newlines, extra spaces and HTML tags removed, lowercased).
2. It is tokenized to 512 tokens and passed to the fine-tuned T5 model.
3. The summary is generated with beam search (4 beams, max 150 tokens).

## Tech stack

Python · PyTorch · Hugging Face Transformers · FastAPI · Uvicorn · Jinja2 · vanilla HTML/CSS/JS
