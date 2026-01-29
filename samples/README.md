# Samples Folder

Example files showing each supported format.

| File | Format | Description |
|------|--------|-------------|
| `sample_qa.txt` | Q&A | Question and Answer pairs |
| `sample_headers.txt` | Headers | Markdown headers with content |
| `sample_numbered.txt` | Numbered | Numbered list items |
| `sample_paragraphs.txt` | Paragraphs | Plain paragraphs |

## Testing

```bash
# Test each format
python convert.py samples/sample_qa.txt -o test.json
python convert.py samples/sample_headers.txt -o test.json  
python convert.py samples/sample_numbered.txt -o test.json
python convert.py samples/sample_paragraphs.txt -o test.json --format paragraphs
```
