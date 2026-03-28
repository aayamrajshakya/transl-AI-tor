# About

[![gitcgr](https://gitcgr.com/badge/aayamrajshakya/transl-AI-tor.svg)](https://gitcgr.com/aayamrajshakya/transl-AI-tor)
This repository contains the code for our CS 5100 Foundations of AI project at Northeastern University (Boston). We're building an NLP-powered web app that translates between English and a low-resource language (Luxembourgish) using a fine-tuned No Language Left Behind (NLLB) model from Hugging Face, with support for direct text input, image uploads via OCR, and audio files, all within an easy-to-use GUI built with Gradio.


> [!IMPORTANT]
> 
> 1. You will need to log in to Hugging Face via the CLI to get a higher download rate limit and to access the FLORES dataset:
>    ```bash
>    huggingface-cli login
>    ```
> 2. Install the required system dependencies **Tesseract OCR** and **FFmpeg** using your OS package manager:
>    ```bash 
>    sudo apt install tesseract-ocr ffmpeg  # Ubuntu/Debian    
>    sudo dnf install tesseract ffmpeg  # Fedora
>    brew install tesseract ffmpeg  # macOS
>    ```


## Group members:
1. Aayam Raj Shakya
2. Brendan Fullerton
3. Abhijeet Khandagale 
