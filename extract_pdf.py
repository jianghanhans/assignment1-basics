import PyPDF2

# 打开PDF文件
pdf_file = open('tests/cs336_spring2025_assignment1_basics.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

# 提取所有页面的文本并保存到文件
with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        f.write(f"Page {page_num + 1}:\n")
        f.write(text)
        f.write("\n" + "="*50 + "\n")

# 关闭文件
pdf_file.close()
print("PDF content saved to pdf_content.txt")