from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from fastapi.templating import Jinja2Templates # UI(to connect with HTML)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# load our model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)

# static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# templating
templates = Jinja2Templates(directory="./templates")

# Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str

# Clean data function
def clean_data(text):
    text = re.sub(r"[\r\n]+", " ", text) # remove lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower() # remove initial spaces

    return text

# Summarization Function
def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean dialogue

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids(tokens)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4, # check 4 possible outputs => take best
        early_stopping=True
    )

    # token ids to text(summary) => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary

# API Endpoints
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")