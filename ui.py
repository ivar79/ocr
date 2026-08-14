import gradio as gr
from inference import OCREngine

ocr = OCREngine()

def ocr_interface(image=None, pdf=None):
    if image is not None:
        result = ocr.read_image(image)
        return result["text"]
    elif pdf is not None:
        result = ocr.read(pdf.name)
        texts = [p["text"] for p in result["pages"]]
        return "\n\n---\n\n".join(texts)
    return "لطفاً یک تصویر یا PDF آپلود کنید."

demo = gr.Interface(
    fn=ocr_interface,
    inputs=[
        gr.Image(type="filepath", label="تصویر"),
        gr.File(label="فایل PDF", file_types=[".pdf"]),
    ],
    outputs=gr.Textbox(label="متن استخراج‌شده", rtl=True),
    title="سیستم OCR فارسی",
    description="متن خود را از تصویر یا PDF استخراج کنید.",
)

if __name__ == "__main__":
    demo.launch(server_port=7860)
