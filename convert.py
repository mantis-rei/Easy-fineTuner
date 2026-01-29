#!/usr/bin/env python3
"""
TEXT TO DATASET CONVERTER
=========================
Converts raw text files into training datasets.

Usage:
    python convert.py input.txt -o dataset.json
    python convert.py input.txt -o dataset.json --format qa
    python convert.py input.txt -o dataset.json --format chunks --chunk-size 500
"""

import argparse
import json
import re
from pathlib import Path

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks

def extract_qa_pairs(text: str) -> list:
    """Extract Q&A pairs from text with common patterns."""
    pairs = []
    
    # Pattern 1: Q: ... A: ...
    pattern1 = re.findall(r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)', text, re.DOTALL | re.IGNORECASE)
    for q, a in pattern1:
        pairs.append({"question": q.strip(), "answer": a.strip()})
    
    # Pattern 2: Question: ... Answer: ...
    pattern2 = re.findall(r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=Question:|$)', text, re.DOTALL | re.IGNORECASE)
    for q, a in pattern2:
        pairs.append({"question": q.strip(), "answer": a.strip()})
    
    # Pattern 3: ### Header followed by content (for markdown)
    pattern3 = re.findall(r'###\s*(.+?)\n(.+?)(?=###|$)', text, re.DOTALL)
    for title, content in pattern3:
        if content.strip():
            pairs.append({"question": f"Explain {title.strip()}", "answer": content.strip()})
    
    # Pattern 4: Numbered items (1. Topic \n Content)
    pattern4 = re.findall(r'\d+\.\s*(.+?)\n(.+?)(?=\d+\.|$)', text, re.DOTALL)
    for topic, content in pattern4:
        if content.strip() and len(content.strip()) > 50:
            pairs.append({"question": f"What is {topic.strip()}?", "answer": content.strip()})
    
    return pairs

def split_by_separator(text: str, separator: str = "\n\n") -> list:
    """Split text by a separator (paragraphs by default)."""
    parts = text.split(separator)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]

def format_as_training_example(content: str, format_type: str = "instruct") -> str:
    """Format content as a training example."""
    if format_type == "instruct":
        return f"### System:\nYou are a helpful assistant.\n\n### User:\nExplain the following:\n\n### Assistant:\n{content}"
    elif format_type == "completion":
        return content
    else:
        return f"### System:\nYou are a helpful assistant.\n\n### User:\nProvide information about this topic.\n\n### Assistant:\n{content}"

def convert_text_to_dataset(
    input_path: str,
    output_path: str,
    format_type: str = "auto",
    chunk_size: int = 500,
    system_prompt: str = "You are a helpful assistant.",
):
    """Convert text file to training dataset."""
    
    print(f"Reading: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"  File size: {len(text):,} characters")
    
    examples = []
    
    # Auto-detect format
    if format_type == "auto":
        if re.search(r'Q:|Question:', text, re.IGNORECASE):
            format_type = "qa"
        elif '###' in text or re.search(r'\d+\.', text):
            format_type = "qa"
        else:
            format_type = "chunks"
        print(f"  Auto-detected format: {format_type}")
    
    # Process based on format
    if format_type == "qa":
        pairs = extract_qa_pairs(text)
        
        if not pairs:
            # Fallback to paragraph splitting
            print("  No Q&A patterns found, using paragraphs...")
            paragraphs = split_by_separator(text)
            for p in paragraphs:
                examples.append({
                    "text": f"### System:\n{system_prompt}\n\n### User:\nExplain this:\n\n### Assistant:\n{p}"
                })
        else:
            for pair in pairs:
                examples.append({
                    "text": f"### System:\n{system_prompt}\n\n### User:\n{pair['question']}\n\n### Assistant:\n{pair['answer']}"
                })
    
    elif format_type == "chunks":
        chunks = split_into_chunks(text, chunk_size)
        
        for i, chunk in enumerate(chunks):
            examples.append({
                "text": f"### System:\n{system_prompt}\n\n### User:\nContinue or explain:\n\n### Assistant:\n{chunk}"
            })
    
    elif format_type == "paragraphs":
        paragraphs = split_by_separator(text)
        
        for p in paragraphs:
            examples.append({
                "text": f"### System:\n{system_prompt}\n\n### User:\nExplain the following:\n\n### Assistant:\n{p}"
            })
    
    elif format_type == "lines":
        lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 20]
        
        for line in lines:
            examples.append({
                "text": f"### System:\n{system_prompt}\n\n### User:\nRespond:\n\n### Assistant:\n{line}"
            })
    
    # Save dataset
    print(f"\nCreated {len(examples)} training examples")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_path}")
    
    # Show sample
    if examples:
        print(f"\n--- Sample Example ---")
        print(examples[0]["text"][:300] + "...")
    
    return examples

def main():
    parser = argparse.ArgumentParser(
        description="Convert text files to training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Formats:
  auto       - Auto-detect Q&A patterns or use chunks
  qa         - Extract Q&A pairs (Q:/A:, Question:/Answer:, ### headers)
  chunks     - Split into overlapping chunks
  paragraphs - Split by empty lines
  lines      - Each line becomes an example

Examples:
  python convert.py book.txt -o dataset.json
  python convert.py notes.txt -o dataset.json --format qa
  python convert.py article.txt -o dataset.json --format chunks --chunk-size 300
  python convert.py faq.txt -o dataset.json --system "You are a trading expert."
        """
    )
    
    parser.add_argument("input", help="Input text file")
    parser.add_argument("-o", "--output", default="dataset.json", help="Output JSON file")
    parser.add_argument("--format", "-f", default="auto", 
                        choices=["auto", "qa", "chunks", "paragraphs", "lines"],
                        help="Conversion format (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Words per chunk (default: 500)")
    parser.add_argument("--system", default="You are a helpful assistant.", 
                        help="System prompt to use")
    
    args = parser.parse_args()
    
    convert_text_to_dataset(
        args.input,
        args.output,
        args.format,
        args.chunk_size,
        args.system,
    )

if __name__ == "__main__":
    main()
