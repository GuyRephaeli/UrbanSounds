# UrbanSounds
* To test the project, first download the UrbanSound8K dataset and unzip it inside the base directory of the project.

## Initial Process
First looking at the dataset and the hint, I realized I should first look at existing approaches for: 
1. Audio classification - Would be useful since this is the problem I was asked to solve. 
2. Transfer learning - The hint basically told me to do it, but also the size of the dataset seemed way too small to be 
the only training data for the model. 
3. Using CNNs in the domain of audio classification - the hint pointed to that direction, but also I previously heard of
state-of-the-art approaches in audio processing that utilize CNNs (WaveNet). 

The first thing I looked up was 'audio classification', and the first result on Google for that search was: 
https://towardsdatascience.com/audio-classification-with-pytorchs-ecosystem-tools-5de2b66e640c.  
This blog post describes a method for converting the audio signal to an image (as a mel-spectrogram), 
and then use a pretrained CNN on that image.   
This basically covers all 3 criteria, and funny enough - the author (pretty sure he is also from Israel) even uses the 
UrbanSound8K dataset to demonstrate the model.  
Another approach I encountered during my search was using SVM on MFCC features extracted from the audio signal. This 
approach is similar to the one presented in the blog in that they both use Mel-frequency as the baseline for the prediction,
but the approach from the blog also utilizes the strength of a pretrained CNN so it might be stronger (and also match the hint).      
That lead me to believe the the approach from the blog post is the way to go, but before committing to that approach, 
I wanted to check some other ANN and CNN models I had heard of for sequential data modeling:
1. TCNs - I have previously heard of TCNs as a state of the art approach for modeling sequential data with CNNS. I ruled
it out because it is designed for sequence2sequence modeling. 
2. Transformers - I heard about Transfomers as a possible replacement for LSTMs. I ruled it out since it is designed 
for attentions models. 
3. TDNNs - I heard about TDNNs in the context of audio classification a few years ago. I ruled it out since it is more 
obscure than CNNs, hence finding pretrained models for transfer learning would be much harder.


I finally decided to go with the model proposed in the first blog post.
## The Data
Before I started to implement, I first wanted to find out some things about the data. Specifically, I wanted to know if
there are any class imbalances or any major differences in the size of the folds. While the folds seemed relatively even,
the classes did not - 8 of the 10 classes had size ~1000 but 2 had size ~400. This imbalance might not be significant,
but maybe oversampling the 2 smaller classes (or undersampling the other 8) might be beneficial during training.
## Implementation - Take 1
First, I wanted a naive implementation of the model described in the blog post, without any modifications. The results 
of that implementation will indicate which changes I would like to introduce to the model.
Since the implementation in the blog post used Pytorch, I decided to also use it to not waste time on the finding the 
equivalent tensorflow functions.  
The first part of the process would be the preprocessing which includes the resampling of the audio, conversion to mono,
transforming it to mel-spectrogram and then clipping/padding the results to a fixed length. Since this operation does
not change between the different folds and each fold requires this process to be run on the entire data-set, I decided 
to run it one time on the entire dataset at the beginning and save the results in a folder next to the audio folder.  
The next part is creating the CNN to classify the spectrograms to the different classes. I chose to use a pretrained 
ResNet18, train it for 6 epochs with batches of size 8 just like the blog post. I had two main difference from the blog:
1. Instead of using SGD with momentum and decreasing learning rate, I used ADAM with a constant learning rate.
2. Instead of training the weights for the entire network, I trained only the weights for the first and last layers.

### Results 
Train Accuracy: `0.7079791973237144`  
Test Accuracy: `0.5668037557822897`  
Test Confusion:  

|                    |Air Conditioner|Car Horn|Children Playing|Dog Bark|Drilling|Engine Idling|Gun Shot|Jackhammer|Siren|Street Music|
| ------------------ | -------------:| ------:| --------------:| ------:| ------:| -----------:| ------:| --------:| ---:| ----------:| 
|**Air Conditioner** |           477 |     11 |             50 |     40 |     76 |         103 |      3 |      127 |  71 |         42 |
|**Car Horn**        |            14 |    196 |             15 |     22 |     69 |          15 |     13 |       25 |  31 |         29 |
|**Children Playing**|           110 |     19 |            534 |     80 |     34 |          29 |      3 |       11 |  47 |        133 |
|**Dog Bark**        |            51 |     28 |             97 |    649 |     35 |          10 |     27 |       10 |  41 |         52 |
|**Drilling**        |           100 |     21 |             23 |     14 |    673 |          27 |     15 |       76 |  34 |         17 |
|**Engine Idling**   |           205 |      7 |             11 |      3 |     63 |         455 |     28 |      153 |  60 |         15 |
|**Gun Shot**        |             2 |      3 |              1 |     16 |     13 |           2 |    318 |        7 |   9 |          3 |
|**Jackhammer**      |           174 |      2 |              2 |      3 |    162 |          54 |     11 |      577 |  11 |          4 |
|**Siren**           |            95 |     26 |             47 |     30 |     40 |          40 |      2 |       14 | 558 |         77 |
|**Street Music**    |            77 |     33 |            150 |     50 |     45 |          23 |      5 |       25 |  98 |        494 |

### Analysis
The first 3 things I noticed:
1. The test accuracy is much lower than the accuracy presented in the blog post.   
2. The train accuracy is much higher than the test accuracy. That may be a result of overfitting.  
3. The model actually performed pretty well on the two smaller classes (Car Horn, Gun Shot).  

The second point may imply that there is some overfitting. The third point implies that there is no need for oversampling 
of the 2 smaller classes.

## Implementation - Take 2
Since the blog presented better performance than what I had achieved, I decided to change my model according to the blog.  
First I decided to use a non-frozen ResNet18 (like the blog post).

### Results  
Train Accuracy: `0.9372974016358862`  
Test Accuracy: `0.6783676014936025`  
Test Confusion: 

|                    |Air Conditioner|Car Horn|Children Playing|Dog Bark|Drilling|Engine Idling|Gun Shot|Jackhammer|Siren|Street Music|
| ------------------ | -------------:| ------:| --------------:| ------:| ------:| -----------:| ------:| --------:| ---:| ----------:| 
|**Air Conditioner** |           416 |      1 |             47 |     51 |    113 |         187 |      2 |       47 |  80 |         56 |
|**Car Horn**        |             2 |    366 |              2 |      8 |      6 |           6 |      5 |        3 |   3 |         28 |
|**Children Playing**|            39 |      2 |            756 |     28 |     18 |          22 |      2 |        3 |  21 |        109 |
|**Dog Bark**        |            21 |     20 |             49 |    850 |      9 |          11 |     10 |        1 |  19 |         10 |
|**Drilling**        |            54 |      3 |             11 |     29 |    700 |          43 |     11 |      102 |  27 |         20 |
|**Engine Idling**   |           151 |     16 |             23 |     14 |    108 |         530 |     18 |      105 |  18 |         17 |
|**Gun Shot**        |             3 |      0 |              2 |      8 |      3 |           1 |    349 |        4 |   3 |          1 |
|**Jackhammer**      |           100 |      5 |              8 |      2 |    169 |          88 |      5 |      547 |  42 |         34 |
|**Siren**           |            37 |      3 |             33 |     20 |     25 |          13 |      0 |        5 | 666 |        127 |
|**Street Music**    |            35 |     14 |             99 |     24 |     28 |          14 |      0 |        8 |  40 |        738 |

### Analysis
Unsurprisingly, the accuracy is now much higher for both train and test, but at the cost of a (slightly) longer run time.  
Still, the train accuracy is much higher than the test accuracy.  
It is now clearer that the model tends to get especially confused between **Air Conditioner** and **Engine Idling**. 
That is understandable since these are very similar sounds.

## Implementation - Take 3
This time I also changed the optimizer to the SGD with momentum from the blog (same parameters). The only difference is
that I did not change the learning rate between epochs as shown in the blog.

### Results
Train Accuracy: `0.9834663766376979`  
Test Accuracy: `0.70292969216996`  
Test Confusion: 

|                    |Air Conditioner|Car Horn|Children Playing|Dog Bark|Drilling|Engine Idling|Gun Shot|Jackhammer|Siren|Street Music|
| ------------------ | -------------:| ------:| --------------:| ------:| ------:| -----------:| ------:| --------:| ---:| ----------:| 
|**Air Conditioner** |           475 |     19 |             72 |     31 |    115 |         128 |      2 |       49 |  60 |         49 |
|**Car Horn**        |             1 |    357 |             13 |     10 |     11 |           7 |      1 |        5 |   2 |         22 |
|**Children Playing**|            27 |      3 |            775 |     30 |     14 |          22 |      0 |        2 |  24 |        103 | 
|**Dog Bark**        |            18 |     16 |             50 |    855 |      6 |           9 |      8 |        0 |  21 |         17 |
|**Drilling**        |            34 |     10 |             42 |     23 |    733 |          13 |      3 |       75 |  44 |         23 |
|**Engine Idling**   |            73 |     11 |             37 |     11 |    146 |         573 |      0 |       96 |  36 |         17 |
|**Gun Shot**        |             3 |      3 |              0 |      5 |      1 |           0 |    355 |        1 |   6 |          0 |
|**Jackhammer**      |            85 |     32 |              4 |      0 |    106 |          48 |      8 |      599 |  50 |         68 |
|**Siren**           |            17 |     10 |             43 |     33 |     63 |          22 |      1 |        1 | 693 |         46 |
|**Street Music**    |            27 |     12 |            143 |     21 |     20 |          14 |      4 |       11 |  34 |        714 |

### Analysis
This model improved upon the last one and was even able to alleviate some of the confusion between **Air Conditioner** 
and **Engine Idling**, but it is still not as good as the model shown in the blog post. That is reasonable since reducing
the learning rate during training can improve a model dramatically.  
On the other hand, the gap between the train accuracy and the test accuracy has increased (the result of better training)
so the overfitting should be addressed.

## Future work
The current model can be further improved by fine-tuning the hyper parameters via greed search.   
Furthermore, fine-tuning the hyper parameters of the preprocessing stage in the same manner would be beneficial as well.  
One could also try different network architectures, different spectrogram transformation and also data augmentation.
More interesting improvements can be achieved by specifically targeting the overfitting and the confusion between 
**Air Conditioner** and **Engine Idling**.  
To address overfitting one can utilize an ensemble model. A very simple and effective way to do so would be bagging.  
To address the confusion one can train a model specifically for **Air Conditioner** and **Engine Idling** and use it 
with the original classifier. For each example that the original classifier thinks it is either **Air Conditioner** or 
**Engine Idling**, the specific classifier will make the final call.

 