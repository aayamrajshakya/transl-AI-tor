import gradio as gr, whisper, pytesseract, warnings, torch
from config import *
from main import initialize_translator, eval_predict, which_device
from PIL import Image


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = which_device() # use the best available device (gpu) or fallback to cpu

# loading models globally once to avoid reloading on every request
tokenizer, model = initialize_translator(LANGUAGE_MODEL)
model.eval()
model = torch.compile(model)
stt_model = whisper.load_model("small", device=device) # speech-to-text model by OpenAI. "small" model should be enough for our purpose


def text_option(text):
    if not text or not text.strip():
        return "Please enter some text."
    translation = eval_predict(tokenizer, model, text, TARGET_LANGUAGE)
    return translation[0]


# https://dev.to/0xkoji/build-a-text-extractor-app-with-python-code-under-30-lines-using-gradio-and-hugging-face-40o7
def img_option(image_path):
    if not image_path:
        return "No image uploaded. Please upload an image.", ""
    
    with Image.open(image_path) as img:
        ocr_output = pytesseract.image_to_string(img).strip()

    if not ocr_output:
        return "No text detected in image.", ""

    chunks = [s.strip() for s in ocr_output.split('\n') if s.strip()]
    translations = eval_predict(tokenizer, model, chunks, TARGET_LANGUAGE)
    translation = ' '.join(translations)
    return ocr_output, translation


# https://medium.com/@verashoda/transcribing-audio-to-text-in-python-using-whisper-290cea2f6090
# https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
def audio_option(audio_path):
    if not audio_path:
        return "No audio provided. Please provide an audio file.", ""
    result = stt_model.transcribe(audio_path)
    transcription = result["text"]
    translation = eval_predict(tokenizer, model, transcription, TARGET_LANGUAGE)
    return transcription, translation[0]


text_tab = gr.Interface(
    fn=text_option,
    inputs=gr.Textbox(lines=7, label="Original text", placeholder="Enter your text here..."),
    outputs=gr.Textbox(lines=7, label="Translated text"),
    flagging_mode="never",
    description=f"Convert {SOURCE_LANGUAGE} text into {TARGET_LANGUAGE}"
)


img_tab = gr.Interface(
    fn=img_option,
    inputs=gr.Image(type="filepath", label="Upload an image"),
    outputs=[gr.Textbox(label="Extracted text"), gr.Textbox(label="Translated text")],
    flagging_mode="manual",
    description=f"Convert {SOURCE_LANGUAGE} text into {TARGET_LANGUAGE}",
    article="Flagging saves a log along with the image, extracted text, and translated text in `.gradio/flagged/..`"
)


audio_tab = gr.Interface(
    fn=audio_option,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Transcribed text"), gr.Textbox(label="Translated text")],
    flagging_mode="manual",
    description=f"Convert {SOURCE_LANGUAGE} text into {TARGET_LANGUAGE}",
    article="Flagging saves a log along with the audio file, transcribed text, and translated text. Try audio samples from https://audio-samples.github.io/"
)


ui = gr.TabbedInterface([text_tab, img_tab, audio_tab], ["Text", "Image", "Audio"], "Transl-AI-tor")
ui.launch()
