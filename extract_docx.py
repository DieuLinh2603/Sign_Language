import sys
import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(docx_path, txt_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for p in tree.findall('.//w:p', ns):
                    text = ''.join(node.text for node in p.findall('.//w:t', ns) if node.text)
                    f.write(text + '\n')
        print(f"Successfully extracted text to {txt_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    extract_text_from_docx(sys.argv[1], sys.argv[2])
