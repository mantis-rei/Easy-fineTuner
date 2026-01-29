# Data Folder

Put your text files (.txt) here for training.

## Supported Formats

### Format 1: Q&A (Recommended)
```
Q: Your question here?
A: Your answer here.

Q: Another question?
A: Another answer.
```

### Format 2: Headers
```
### Topic Name
Content explaining the topic...

### Another Topic
More content here...
```

### Format 3: Numbered
```
1. Topic Name
Explanation of the topic...

2. Another Topic
More explanation...
```

### Format 4: Paragraphs
```
First paragraph with information...

Second paragraph with more info...
```

## Usage

1. Put your .txt files in this folder
2. Run: `python batch_convert.py`
3. Train: `python train.py --model MODEL --dataset dataset.json`
