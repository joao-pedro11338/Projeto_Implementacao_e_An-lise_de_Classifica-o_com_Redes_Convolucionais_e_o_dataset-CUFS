# Projeto_Implementacao_e_An-lise_de_Classifica-o_com_Redes_Convolucionais_e_o_dataset-CUFS

## Descrição do Projeto
Este projeto utiliza técnicas de aprendizado profundo para construir um modelo de classificação de rostos baseado em uma Rede Neural Convolucional (CNN). Todo o processo do desenvolvimento da atividade pode ser encontrado no pdf 'RELATÓRIO TÉCNICO: Implementação e Análise de Classificação com Redes Convolucionais e o dataset CUFS'

## Índice

- [Descrição do Projeto](#descrição-do-projeto)
- [Objetivo do Projeto ](#objetivo-do-projeto)
- [Instruções para Execução do Código](#instruções-para-execução-do-código)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Principais Conclusões e Considerações](#principais-conclusões-e-considerações)
## Objetivo do Projeto
O objetivo deste projeto é implementar e avaliar um modelo de rede neural convolucional (CNN) para a tarefa de classificação de faces de pessoas, utilizando o dataset CUHK Face Sketch Database (CUFS). Este dataset contém imagens de rostos reais e seus respectivos esboços, e o objetivo é treinar um modelo capaz de classificar corretamente essas imagens. Para isso, foi desenvolvido um modelo autoral de CNN, que inclui etapas de pré-processamento, construção do modelo, treinamento e avaliação. A expectativa é que o modelo consiga identificar e classificar as imagens com boa acurácia, mesmo diante de possíveis desafios, como a qualidade das imagens ou a diferença entre rostos reais e esboços.

## Instruções para Execução do Código
1. Garanta que as dependências estejam instaladas. O projeto exige Python 3.x, junto com as bibliotecas TensorFlow 2.x, Keras, OpenCV e Matplotlib. Use o comando:
```bash
   pip install tensorflow opencv-python matplotlib
```

2. Faça o download do dataset CUHK Face Sketch Database (CUFS) link:(*https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs*) e garantir que as imagens estejam organizadas de forma adequada, com pastas separando as imagens dos rostos reais e dos esboços. No código, é necessário alterar a variável DATASET_PATH para o local correto onde o dataset foi armazenado. Com isso, o código pode ser executado diretamente no ambiente de desenvolvimento com o comando python nome_do_arquivo.py.
  
3. Durante a execução, o código irá treinar o modelo, realizar a validação e gerar gráficos de acurácia e perda, que mostram o desempenho do modelo em cada época de treinamento.

## Tecnologias utilizadas
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3C6A?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-0076A8?style=flat&logo=matplotlib&logoColor=white)


## Principais Conclusões e Considerações
O modelo de rede neural convolucional obteve bons resultados de treinamento, com acurácia próxima de 1.0, porém apresentou variações durante a validação. Esses picos e quedas indicam que o modelo pode estar sofrendo de overfitting, ou seja, ajustando-se demais aos dados de treinamento e não conseguindo generalizar bem para novos dados. Isso ficou evidente pela diferença entre o desempenho no treinamento e no conjunto de validação, onde a acurácia do treinamento foi muito alta, mas a validação teve flutuações significativas.
Um fator que pode ter impactado o desempenho do modelo é a limitação do próprio dataset, que possui imagens com variações significativas de iluminação e ângulos, o que torna a tarefa de classificação mais difícil. Além disso, a quantidade limitada de dados de treinamento pode ter dificultado a capacidade de generalização do modelo, o que é um desafio comum em tarefas de aprendizado de máquina.
Como sugestão para melhorias futuras, seria interessante explorar técnicas de data augmentation para aumentar a diversidade dos dados de treinamento. Isso ajudaria a melhorar a robustez do modelo e reduzir o risco de overfitting. Outra possível melhoria seria o uso de redes pré-treinadas (transfer learning), que podem trazer melhores resultados ao aproveitar o aprendizado de modelos mais complexos treinados em grandes bases de dados.
