import gradio as gr
from config import *
from main import initialize_translator,translate_text

device = which_device() # use the best available device (gpu) or fallback to cpu

tokenizer, model = initialize_translator(LANGUAGE_MODEL)
model.to(device)
model.eval()

def translate_func(english_txt):
    translation = translate_text(tokenizer, model, english_txt, TARGET_LANGUAGE)
    return translation

gr.Interface(
    fn=translate_func,
    inputs=[gr.Textbox(lines=4, placeholder="Enter your text here...")],
    outputs="text",
    title="Transl-AI-tor",
    description=f"Convert {SOURCE_LANGUAGE} text into {TARGET_LANGUAGE}").launch(debug=True)