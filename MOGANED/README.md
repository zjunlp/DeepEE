# MOGANED
An unofficial pytorch code reproduction of EMNLP-19 paper "Event Detection with Multi-Order Graph Convolution and Aggregated Attention"


## Prerequisites

1. Prepare **ACE 2005 dataset**.(You can get ACE2005 dataset here: https://catalog.ldc.upenn.edu/LDC2006T06) 

2. Use [nlpcl-lab/ace2005-preprocessing](https://github.com/nlpcl-lab/ace2005-preprocessing) to preprocess ACE 2005 dataset in the same format as the [data/sample.json](https://github.com/ll0iecas/MOGANED/blob/master/data/sample.json). 

## Usage

### Preparations

1、put the processed data into ./data, or you can modify path in constant.py. 
  
2、put word embedding file into ./data, or you can modify path in constant.py. (You can download GloVe embedding here: https://nlp.stanford.edu/projects/glove/)

### Train
```
python train.py
```

All network and training parameters are in [constant.py](https://github.com/ll0iecas/MOGANED/blob/master/consts.py). You can modify them in your own way.

About the word embedding, we found that wordemb in the way (train the word embedding using Skip-gram algorithm on the NYT corpus) got better performance than the [glove.6B.100d](https://nlp.stanford.edu/projects/glove/). So we choose 100.utf8 (you can get it here https://github.com/yubochen/NBTNGMA4ED) as our word embedding vector.

## Result	

### Performance	

<table>	
  <tr>	
    <th rowspan="2">Method</th>	
    <th colspan="3">Trigger Classification (%)</th>	
  </tr>	
  <tr>	
    <td>Precision</td>	
    <td>Recall</td>	
    <td>F1</td>	
  </tr>	
  <tr>	
    <td>MOGANED(original paper)</td>	
    <td>79.5</td>	  
    <td>72.3</td>	
    <td>75.7</td>	
  </tr>	
  <tr>	
    <td>MOGANED(this code)</td>	
    <td>78.8</td>
    <td>72.3</td>	
    <td>75.4</td>	
  </tr>	
</table>	

## Note

  In many cases, the trigger is a phrase. Therefore, we treat consecutive tokens which share the same predicted label as a whole trigger. So we don't use BIO schema for trigger word. This strategy comes from "Exploring Pre-trained Language Models for Event Extraction and Generation" (ACL 2019), Yang et al. [[paper]](https://www.aclweb.org/anthology/P19-1522.pdf)

## Reference

* Event Detection with Multi-Order Graph Convolution and Aggregated Attention (EMNLP 2019), Yan et al. [[paper]](https://www.aclweb.org/anthology/D19-1582.pdf)
* Nlpcl-lab's bert_event_extraction repository [[github]](https://github.com/nlpcl-lab/bert-event-extraction)
* lx865712528's EMNLP2018-JMEE repository [[github]](https://github.com/lx865712528/EMNLP2018-JMEE)
