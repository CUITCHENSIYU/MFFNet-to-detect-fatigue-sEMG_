# MFFNet-to-detect-fatigue-sEMG_
##The purpose of the experiment
The main content of this warehouse comes from a work in our laboratory: using multi-dimensional feature fusion network to analyze sEMG data and detect muscle fatigue.

##Dependent environment
* torch -->1.1.0
* numpy -->1.17
* tenSorboard -->1.7
* cuda -->10.0
* torchsummary -->1.5.1

## File introduce

* tools.py --> Custom loss function。
* DataHelper.py --> load train data and test data: numpy->torch.tensor
* train.py --> model train script
* Note --> Other documents(include model file) will be announced after the paper is published.

## Datasets
* Dataset1 --> is the data of our laboratory, if you need it, you can contact us by email:[address](1351146953@qq.com).
Of course, it will be provided after the paper is published.
* Dataset2 --> is provided by Michalis et al[1]. and can be obtained through the contact 
method provided in the paper.

## Data preprocessing
Here we show the processing results of dataset 1

* Filter

![1-D](https://github.com/CUITCHENSIYU/MFFNet-to-detect-fatigue-sEMG_/tree/master/images/1-D.png)

* STFT

![2-D](https://github.com/CUITCHENSIYU/MFFNet-to-detect-fatigue-sEMG_/tree/master/images/2-D.png)

## Result
Here we show some results of the experiment

![result](https://github.com/CUITCHENSIYU/MFFNet-to-detect-fatigue-sEMG_/tree/master/images/result.png)

## Reference
[1] M. Papakostas, V. Kanal, M. Abujelala, K. Tsiakas, F. Makedon, Physical
fatigue detection through EMG wearables and subjective user reports: a
machine learning approach towards adaptive rehabilitation, in: Proceedings
450 of the 12th ACM International Conference on PErvasive Technologies Related
to Assistive Environments, PETRA 2019, Island of Rhodes, Greece,
June 5-7, 2019, pp. 475-481.
