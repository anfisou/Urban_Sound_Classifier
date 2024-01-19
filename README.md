# English Version

This project was developed for the subject 'Aprendizagem Computacional II' by the students [André Sousa](https://github.com/anfisou), [Inês Cardoso](https://github.com/LilttleTurtle) and [Paulo Silva](https://github.com/WrekingPanda).

### The project

In this project we received a dataset, the urbansound8k dataset, that contains 8732 labeled sound files, each with a duration less than or equal to 4 seconds.  
These sounds excerpt are labeled according to the following classes:
- Air conditioner
- Car horn
- Children playing
- Dog bark
- Drilling
- Engine idling
- Gun shot
- Jackhammer
- Siren
- Street music
  
The aim of this project is to build classifiers that are able to determine to which of the previous 10 classes a given, unseen, sound excerpt belongs to.

### Notebooks

- Feature_Extraction.ipynb: In this notebook we looked at some features possible to extract from audio files and extracted the ones that will be used later on as inputs for the model (mfccs,chroma_stft);

- CNN.ipynb: A Convolutional Neural Network (CNN or ConvNet) is a type of neural network that has a specific architecture with layers designed to automatically and adaptively learn hierarchical patterns and representations.

- RNN.ipynb: A Recurrent Neural Network (RNN) is a type of artificial neural network designed for sequential data processing. Different from traditional feedforward neural networks, which process input data in a single pass, RNNs have connections that form directed cycles, allowing for them to maintain a hidden state that captures information about previous inputs. The presence of this hidden state allows RNNs to demonstrate temporal dynamics, making them well-suited for tasks that entail sequences, for example time series analysis, natural language processing, and speech recognition.

### Notes

- The following versions were used 
   * python 3.9.6
   * numpy 1.23.0
   * pandas 2.1.1
   * seaborn 0.13.0
   * matplotlib 3.6.0
   * librosa 0.10.1
   * sklearn 0.0.post10
   * tensorflow 2.14.0
   * keras 2.14.0
   * pickle 4.0

- DeepFool: The DeepFool algorithm is an adversarial attack method designed to disturb the  input data in a way that misleads deep neural networks into making incorrect predictions. The goal of this algorithm is to generate small, imperceptible perturbations to input samples that can cause a deep neural network to change their prediction. 
In our project we used the method on both of the network models.

![deepfool](images/deepfool.png)

With original inputs, the RNN predicted dog bark

![mfcc_original](images/mfcc_original.png) ![chromagram_original](images/chromagram_original.png)

With slightly altered inputs, the RNN predicts street music 

![mfcc_alterado](images/mfcc_alterado.png) ![chromagram_alterado](images/chromagram_alterado.png)

Thanks to DeepFool, it is clear that, for this specific example of learning model and sound sample, the mfccs features barelly change. With this we can assume that they have a relatively low relevance to the model and it would probabily be beneficial to use another feature as input.


### References:

- [https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)

- [https://www.kaggle.com/code/badl071/urban-sounds-classification-using-cnns](https://www.kaggle.com/code/badl071/urban-sounds-classification-using-cnns)

- [https://github.com/AmritK10/Urban-Sound-Classification](https://github.com/AmritK10/Urban-Sound-Classification)

- [https://github.com/aminul-huq/DeepFool/blob/master/DeepFool.ipynb](https://github.com/aminul-huq/DeepFool/blob/master/DeepFool.ipynb)

- [https://openaccess.thecvf.com/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf)


# Versão Portuguesa

Este projeto foi desenvolvido no âmbito da unidade curricular 'Aprendizagem Computacional II' pelos alunos [André Sousa](https://github.com/anfisou), [Inês Cardoso](https://github.com/LilttleTurtle) e [Paulo Silva](https://github.com/WrekingPanda).

### O Projeto

Neste projeto, recebemos o conjunto de dados urbansound8k, que contém 8732 ficheiros de áudio rotulados, cada um com duração inferior ou igual a 4 segundos.
Esses excertos de som são classificados de acordo com as seguintes classes:

- Ar condicionado
- Buzina de carro
- Crianças brincando
- Cão a ladrar
- Máquina de perfuração
- Motor 
- Tiro de arma
- Martelo pneumático
- Sirene
- Música de rua

O objetivo deste projeto é construir modelos capazes de determinar a qual das 10 classes mencionadas anteriormente pertence um excerto de som, não visto anteriormente.

### Notebooks

- Feature_Extraction.ipynb: Neste caderno, examinamos algumas características possíveis de extrair de arquivos de áudio e extraímos aquelas que serão usadas posteriormente como inputs para o modelo (mfccs, chroma_stft);

- CNN.ipynb: Uma Rede Neural Convolucional (CNN ou ConvNet) é um tipo de rede neural que possui uma arquitetura específica com camadas projetadas para aprender automaticamente padrões e representações hierárquicas de forma adaptativa.

- RNN.ipynb: Uma Rede Neural Recorrente (RNN) é um tipo de rede neural artificial projetada para o processamento de dados sequenciais. Diferentemente das redes neurais tradicionais de feedforward, que processam dados de entrada em uma única passagem, as RNNs têm conexões que formam ciclos direcionados, permitindo-as manter um estado oculto que captura informações sobre entradas anteriores. A presença desse estado oculto permite que as RNNs representem dinâmicas temporais, tornando-as adequadas para tarefas que envolvem sequências, como análise de séries temporais, processamento de linguagem natural e reconhecimento de fala.

### Notas

- Foram usadas as seguintes versões: 
   * python 3.9.6
   * numpy 1.23.0
   * pandas 2.1.1
   * seaborn 0.13.0
   * matplotlib 3.6.0
   * librosa 0.10.1
   * sklearn 0.0.post10
   * tensorflow 2.14.0
   * keras 2.14.0
   * pickle 4.0

- DeepFool: O algoritmo DeepFool é um método de ataque adversarial projetado para perturbar os dados de input de uma maneira que induza as redes neurais a fazer previsões incorretas. O objetivo deste algoritmo é gerar pequenas perturbações imperceptíveis nas amostras de entrada que podem fazer com que uma rede neural as classifique  de outra forma. Neste projeto, o método foi testado em ambos os modelos de rede.

![deepfool](images/deepfool.png)

Com os inputs originais, a RNN classificava com cão a ladrar

![mfcc_original](images/mfcc_original.png) ![chromagram_original](images/chromagram_original.png)

Com os inputs ligeiramente alterados, a RNN já classifica música de rua

![mfcc_alterado](images/mfcc_alterado.png) ![chromagram_alterado](images/chromagram_alterado.png)

Graças ao DeepFool, fica claro que, para este modelo específico e amostra de som, as características mfccs mal mudam. Com isso, podemos assumir que elas têm uma relevância relativamente baixa para o modelo e provavelmente seria benéfico usar outra tipo de método de extração como input.

### Referências:

- [https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)

- [https://www.kaggle.com/code/badl071/urban-sounds-classification-using-cnns](https://www.kaggle.com/code/badl071/urban-sounds-classification-using-cnns)

- [https://github.com/AmritK10/Urban-Sound-Classification](https://github.com/AmritK10/Urban-Sound-Classification)

- [https://github.com/aminul-huq/DeepFool/blob/master/DeepFool.ipynb](https://github.com/aminul-huq/DeepFool/blob/master/DeepFool.ipynb)

- [https://openaccess.thecvf.com/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf)
