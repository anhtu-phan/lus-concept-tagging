# lus-concept-tagging

## How to run
### Requirements
[OpenFST](http://www.openfst.org/twiki/bin/view/FST/WebHome)

[OpenGRM](http://www.opengrm.org/twiki/bin/view/GRM/NGramLibrary)

python3.6
### Install 
    pip install -r requirements.txt

### Run

    python concept_tagging_wfsm.py 
            [--data_type='_norm_no_stop_word'] \
            [--ngram_order_lm=1] \
            [--ngram_order_wt=2] \
            [--smooth_method_lm='witten_bell']
            [--smooth_method_wt='witten_bell']

- `data_type`: type of runing dataset
    + `''`: original dataset
    + `'_norm_no_stop_word'`: removing stop words and applying normalization
- `ngram_order_lm`: ngram order of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{LM_1})
- `ngram_order_wt`: ngram order of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{W2T_{MLE}})
- `smooth_method_lm`: smooth method of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{LM_1}):
    + `'witten_bell'`
    + `'absolute'`
    + `'katz'`
    + `'kneser_ney'`
    + `'presmoothed'`
    + `'unsmoothed'`
- `smooth_method_wt`: smooth method of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{W2T_{MLE}}):
    + `'witten_bell'`
    + `'absolute'`
    + `'katz'`
    + `'kneser_ney'`
    + `'presmoothed'`
    + `'unsmoothed'`
    
See [report.pdf](report.pdf) for more information
