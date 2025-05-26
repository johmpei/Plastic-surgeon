
import fitz  # PyMuPDF
import os

pdf_path = "足爪治療マスターbook.pdf"
output_dir = "converted_text"

os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)
for i, page in enumerate(doc):
    text = page.get_text()
    with open(f"{output_dir}/page_{i+1:03}.txt", "w", encoding="utf-8") as f:
        f.write(text)

print("✅ PDFをテキストファイルに変換完了！")
