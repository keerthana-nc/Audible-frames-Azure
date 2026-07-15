# Eval Dataset

**Source:** MS COCO 2017 validation set  
**Size:** 30 images  
**Captions per image:** 5 human-written reference captions  

## Citation

```
Microsoft COCO: Common Objects in Context
Tsung-Yi Lin, Michael Maire, Serge Belongie, et al.
European Conference on Computer Vision (ECCV), 2014.
https://arxiv.org/abs/1405.0312
https://cocodataset.org
```

## Setup

Images and captions are downloaded by running:

```bash
python evals/prepare_dataset.py
```

The `images/` folder and `captions.json` file are not committed to git
(they're gitignored) because the images are copyrighted by their original
photographers and licensed under Creative Commons. Always cite COCO when
using this data.
