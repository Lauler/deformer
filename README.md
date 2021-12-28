## DeFormer: correcting the usage of 'de' and 'dem' in the Swedish language

In modern spoken Swedish, the pronouns "de" and "dem" are pronounced the same way ("dom"). However, in written Swedish we still distinguish between the object form (dem) and the subject form (de) of third person plural pronouns.

Since they are homophones, an increasing number of Swedes have begun to confuse their usage, similar to "their" vs "they're", or "your" vs "you're" in English.

DeFormer is a transformer language model based on the Swedish [KB-BERT](https://arxiv.org/abs/2007.01658). It has been trained to perform token classification. Each word piece (token) is classified in to one of three categories:

1. **`ord`** (all background words which are not de/dem belong to this category) 
2. **`DE`**
3. **`DEM`**

## Training data

DeFormer has been trained on sentences from the European Parliament and Swedish Wikipedia. These were downloaded from [OPUS](https://opus.nlpl.eu/) (the Open Parallel Corpus project). The data sources were selected because they were presumed to have correct language usage. 

Only sentences containing `de` or `dem` -- or both -- were kept in the creation of the training dataset. In the table below, we present summary statistics of sentences that were kept after filtering the data.


| Source                                                                                           | Sentences   |  # de       | # dem       | de/dem ratio |
| -----------                                                                                      | ----------- | -------     | -------     | ------------ |
| [Europaparl sv.txt.gz](https://opus.nlpl.eu/download.php?f=Europarl/v8/mono/sv.txt.gz)           | 500660      | 465977      | 54331       | 8.57x        |
| [JRC-Acquis raw.sv.gz](https://opus.nlpl.eu/download.php?f=JRC-Acquis/mono/JRC-Acquis.raw.sv.gz) | 417951      | 408576      | 17028       | 23.99x       |
| [Wikimedia sv.txt.gz](https://opus.nlpl.eu/download.php?f=wikimedia/v20210402/mono/sv.txt.gz)    | 630601      | 602393      | 38852       | 15.48x       |
| **Total**                                                                                        | **1549212** | **1476946** | **110211**  | **13.40x**   |

In the training, random substitutions were introduced where `de` or `dem` were excahnged against the opposite word. The DeFormer model was then challenged to classify which of the previously mentioned 3 categories a word belonged to. 

## Accuracy 

DeFormer was evaluated on a held out validation set of 31200 sentences from the above data sources. Random substitutions were introduced here as well to challenge the model. The table below displays the accuracy of the model. A significant proportion of the erroneous predictions were in the form of "`de/dem som`-constructions. These are ambiguous cases in Swedish and are not viewed as errors, since [both forms are equally accepted](https://www4.isof.se/cgi-bin/srfl/visasvar.py?sok=dem%20som&svar=79718&log_id=705355). 

|             | Accuracy    |
| ----------- | ----------- |
| de          | 99.9\%      |
| dem         | 98.6\%      |


## Instructions to train the model

Download the data. If you are on a Linux/Mac system, you can run `bash download_data.sh`. This will create a `data/` folder and download the relevant datasets. Windows users can check the same file and download the data files manually from the links in the file. 

Secondly, run `create_dedem_dataset.py`. This extracts the sentences containing `de/dem` from each respective dataset and creates a file called `dedem_corpus.feather`. 

Third, run `train.py` to train the model. 