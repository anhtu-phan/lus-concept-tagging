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
            [--alg_type='mle'] \
            [--data_type='_norm_no_stop_word'] \
            [--ngram_order_lm=1] \
            [--ngram_order_wt=2] \
            [--ngram_order_joint=2] \
            [--smooth_method_lm='witten_bell'] \
            [--smooth_method_wt='witten_bell'] \
            [--smooth_method_joint='witten_bell']
- `alg_type`: type of model
    + `mle`: Maximum Likelihood Estimation (Emission Probabilities)
    + `joint`: Joint Distribution Modeling
- `data_type`: type of runing dataset
    + `''`: original dataset
    + `'_norm_no_stop_word'`: removing stop words and applying normalization
- `ngram_order_lm`: ngram order of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{LM_1})
- `ngram_order_wt`: ngram order of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{W2T_{MLE}})
- `ngram_order_joint`: ngram order of joint model
- `smooth_method_lm`: smooth method of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{LM_1}):
    + `'witten_bell'`
    + `'absolute'`
    + `'katz'`
    + `'kneser_ney'`
    + `'presmoothed'`
    + `'unsmoothed'`
- `smooth_method_wt`: smooth method of ![](https://latex.codecogs.com/png.image?\dpi{80}&space;\inline&space;\lambda_{W2T_{MLE}}):
    + same choice as `smooth_method_lm`
- `smooth_method_joint`: smooth method of joint model 
    + same choice as `smooth_method_lm`
- The results in `result` folder with file name `{alg_type}_{data_type}_[{smooth_method_lm}_{smooth_method_wt}_{ngram_order_lm}_{ngram_order_wt}][{smooth_method_joint}_{ngram_order_joint}]`

See [report.pdf](report.pdf) for more information
