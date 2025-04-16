## 🇧🇷 Modelos de Classificação de Texto em Português com BERTimbau

Este repositório contém scripts para treinar e avaliar modelos de classificação de texto em português com base na arquitetura **BERTimbau**. O foco está em tarefas como:

- ✅ **Análise de Sentimento**  
- ✅ **Detecção de Discurso de Ódio**  
- ✅ **Identificação de Linguagem Imprópria**

Todos os modelos desenvolvidos aqui foram aplicados na seguinte aplicação prática:  
🔗 [YouTubeSafeKids-Python](https://github.com/Louzada27/YouTubeSafeKids-Python)

---

## 📑 Sobre o Projeto

Este projeto fornece um framework completo para realizar **fine-tuning** do modelo [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased) (BERTimbau) em tarefas de **classificação de texto** no idioma português.  
Foram utilizados os seguintes datasets:

- **LexiconPT** (para Análise de Sentimento)
- **HateBR** (para Detecção de Discurso de Ódio)
- **OLID-BR** (para Identificação de Linguagem Imprópria)

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **PyTorch**: Framework de deep learning
- **Hugging Face Transformers**: Biblioteca para modelos de PLN de última geração
- **Hugging Face Datasets**: Biblioteca para carregar e processar datasets
- **Scikit-learn**: Para métricas de avaliação e funções utilitárias (como pesos de classe)
- **Pandas**: Para manipulação de dados
- **NumPy**: Para operações numéricas

---

## 📁 Estrutura do Projeto

.
├── Data-set/                     # Diretório contendo os datasets
│   ├── HateBR-main/              # Arquivos do dataset HateBR
│   ├── lexiconPT-master/         # Arquivos do dataset LexiconPT
│   └── ...                       # Outros datasets que podem ser adicionados
├── train_sentiment.py            # Script para treinar modelo de análise de sentimento (LexiconPT)
├── train_hatbr.py                # Script para treinar modelo no dataset HateBR
├── train_olid.py                 # Script para treinar modelo no dataset OLID-BR
├── train_model.py                # Script genérico para fine-tuning do BERTimbau (potencialmente menos usado agora)
├── test_sentimento.py            # Script para testar o modelo de sentimento
├── test_hatbr.py                 # Script para testar o modelo HateBR
├── test_olidBr.py                # Script para testar o modelo OLID-BR
├── README.md                     # Este arquivo
└── ...                           # Outros arquivos de configuração/utilitários

---

## 🚀 Como Usar

### 1. Configuração

a. **Clone o repositório:**
    ```bash
    git clone <url-do-repositorio>
    cd <diretorio-do-repositorio>
    ```

b. **Instale as dependências:** É recomendado usar um ambiente virtual.
    ```bash
    pip install torch transformers datasets pandas numpy scikit-learn
    ```
    *(Você pode precisar de versões específicas com base na compatibilidade. Verifique os imports nos scripts se ocorrerem erros.)*

c. **Baixe os Datasets:**
- Link para o dataset **HateBR**: https://github.com/franciellevargas/HateBR
- Link para o dataset **LexiconPT**: https://github.com/sillasgonzaga/lexiconPT
- Link para o dataset **OLID-BR**: https://huggingface.co/datasets/dougtrajano/olid-br

---

### 2. Treinando Modelos

#### HateBR

🧠 **Fine-Tuning BERTimbau para Detecção de Toxicidade com HateBR**

Este repositório contém um script em Python para realizar fine-tuning de modelos BERTimbau no dataset **HateBR**, que possui comentários em português rotulados com diferentes níveis de toxicidade:

- 0: Não tóxico
- 1: Tóxico
- 2: Discurso de ódio

✨ **Funcionalidades:**

- Pré-processamento automático do dataset HateBR
- Tokenização com **BertTokenizer**
- Treinamento com **BertForSequenceClassification** da biblioteca 🤗 **Transformers**
- Cálculo de métricas por classe (F1, precisão, recall, acurácia)
- Suporte a **early stopping** e **TensorBoard**
- Totalmente compatível com **CPU**

#### OLYD-BR
# Fine-Tuning com BERTimbau para Detecção de Toxicidade (OLID-BR)

Este projeto realiza o **fine-tuning** de modelos BERT em português (como o BERTimbau) para **classificação multilabel de toxicidade** utilizando o dataset **OLID-BR**, uma versão brasileira do OLID (Offensive Language Identification Dataset).

---

## 📦 Funcionalidades

- Treinamento de modelos BERT para detecção de múltiplas categorias de toxicidade
- Cálculo de métricas detalhadas (`macro`, `micro`, por classe, etc.)
- Aplicação de thresholds específicos por classe
- Salvamento de logs e métricas de avaliação
- Suporte a pesos customizados (`pos_weight`) para lidar com desbalanceamento

---

## 🧠 Classes de Toxicidade

O modelo é treinado para identificar as seguintes classes:

- `health`
- `ideology`
- `insult`
- `lgbtqphobia`
- `other_lifestyle`
- `physical_aspects`
- `profanity_obscene`
- `racism`
- `sexism`
- `xenophobia`
## 🛠️ Técnicas de Treinamento Utilizadas

O modelo foi treinado com as seguintes abordagens e práticas:

- ✅ **Modelo pré-treinado BERTimbau**: `neuralmind/bert-base-portuguese-cased`
- ✅ **Classificação multilabel**: usando função de perda `BCEWithLogitsLoss`
- ✅ **Pesos por classe (`pos_weight`)**: aplicados para lidar com o desbalanceamento entre classes
- ✅ **Tokenização com `AutoTokenizer`**: preservando a estrutura linguística do português
- ✅ **Thresholds personalizados**: valores ajustados por classe para melhorar a performance multilabel
- ✅ **Estratégia de divisão do dataset**: treino, validação e teste já balanceados via `datasets` HuggingFace
- ✅ **Aprimoramento com `EarlyStoppingCallback`**: para evitar overfitting com `patience=3`
- ✅ **Avaliação com métricas multilabel**: incluindo micro/macro F1, Jaccard Score, Hamming Loss
- ✅ **Uso do HuggingFace `Trainer`**: para facilitar e padronizar o loop de treinamento e avaliação

Essas técnicas foram escolhidas para maximizar a performance em tarefas de **classificação multilabel com dados desbalanceados** — um cenário comum em detecção de discurso tóxico.

Apesar do foco do modelo ser em toxidade, ele se demostru bastante efdicaz em indetifdcaçção de liguamge improria em textos, pricipal,emte se asscoiado como Regex par o auxlio da tarefa.

#### LEXICON-PT

# Análise de Sentimento com BERTimbau e LexiconPT 🇧🇷

Este projeto treina um modelo BERT em português para **classificação de sentimentos** (positivo, negativo, neutro) com base no dataset **LexiconPT**.

---

## 📌 Objetivo

Realizar fine-tuning de um modelo BERTimbau (`neuralmind/bert-base-portuguese-cased`) para identificar **polaridade semântica** em palavras/tweets do português brasileiro, utilizando um conjunto de dados lexical.

---

## 🛠️ Técnicas de Treinamento Utilizadas

O modelo foi treinado com diversas boas práticas de NLP e deep learning:

- ✅ **Modelo pré-treinado BERTimbau** (`neuralmind/bert-base-portuguese-cased`)
- ✅ **Classificação de sentimento com 3 classes**: Positivo, Negativo, Neutro
- ✅ **Uso de `CrossEntropyLoss` com pesos de classe** (`class_weights`) para lidar com desbalanceamento de classes
- ✅ **Tokenização com `BertTokenizer`**, preservando a estrutura do idioma
- ✅ **Uso de `Trainer` personalizado (`WeightedTrainer`)** com override de `compute_loss`
- ✅ **Early Stopping** com `EarlyStoppingCallback(patience=3)` para evitar overfitting
- ✅ **Divisão estratificada do dataset** (80% treino, 20% teste)
- ✅ **Avaliação com métricas robustas**: F1, precisão, recall, acurácia
- ✅ **Pré-processamento automático com padding, truncamento e `max_length` configurável**
- ✅ **Treinamento com suporte a `fp16` (mixed precision) para maior eficiência**

---

## 📁 Dataset

O projeto utiliza o dataset [LexiconPT (Oplexicon v3.0)](https://github.com/rafjaa/lexiconPT), um dicionário de polaridade semântica de palavras em português.

- **Fonte**: CSV localizado em `app/nlp/Data-set/lexiconPT-master/data/csv/oplexicon_v3.0.csv`
- **Colunas utilizadas**: `term`, `polarity`
- **Rótulos mapeados**:
  - `1 → 0` (Positivo)
  - `-1 → 1` (Negativo)
  - `0 → 2` (Neutro)



## 3. Avaliando Modelos
### 1. Modelo de Análise de Sentimento (BERTimbau + LexiconPT)

O modelo de análise de sentimento foi treinado para classificar textos em três categorias: **Positivo**, **Negativo** e **Neutro**, utilizando o dataset **LexiconPT**.

### Métricas Gerais:
- **Acurácia**: 76.84%
- **Precisão**: 76.75%
- **Revocação**: 76.84%
- **F1-Score**: 76.63%

### Matriz de Confusão:
| Predito:     | Positivo | Negativo | Neutro |
|--------------|----------|----------|--------|
| **Real: Positivo**  | 1052     | 292      | 360    |
| **Real: Negativo**  | 118      | 2586     | 254    |
| **Real: Neutro**    | 272      | 195      | 1309   |

### Desempenho por Classe:
- **Positivo**:
  - **Precisão**: 72.95%
  - **Revocação**: 61.74%
  - **F1-Score**: 66.88%
- **Negativo**:
  - **Precisão**: 84.15%
  - **Revocação**: 87.42%
  - **F1-Score**: 85.76%
- **Neutro**:
  - **Precisão**: 68.07%
  - **Revocação**: 73.70%
  - **F1-Score**: 70.78%

---

## 2. Modelo de Classificação de Toxicidade (BERTimbau + HateBR)

Este modelo foi projetado para classificar conteúdos como **Tóxicos** ou **Não Tóxicos**, utilizando o dataset **HateBR**.

### Métricas Gerais:
- **Acurácia**: 96.71%
- **Precisão**: 96.72%
- **Revocação**: 96.71%
- **F1-Score**: 96.71%

### Matriz de Confusão:
| Predito:      | Não Tóxico | Tóxico |
|---------------|------------|--------|
| **Real: Não Tóxico** | 3369       | 131    |
| **Real: Tóxico**    | 99        | 3401   |

### Desempenho por Classe:
- **Não Tóxico**:
  - **Precisão**: 97.15%
  - **Revocação**: 96.26%
  - **F1-Score**: 96.70%
- **Tóxico**:
  - **Precisão**: 96.29%
  - **Revocação**: 97.17%
  - **F1-Score**: 96.73%

---

## 3. Detecção de linguagem impropria (BERTimbau + OLYD-BR)


## Métricas Gerais

- **Loss de Avaliação**: 0.1867
- **Acurácia**: 51.44%
- **F1-Score (Média Ponderada)**: 74.30%
- **Micro F1-Score**: 77.83%
- **Macro F1-Score**: 35.69%
- **Precisão (Média Ponderada)**: 76.98%
- **Revocação (Média Ponderada)**: 74.63%
- **Hamming Loss**: 0.0661
- **Jaccard Score (Média Ponderada)**: 63.01%

---

## Métricas por Classe

A seguir, as métricas detalhadas de F1-Score, Precisão e Revocação para cada classe.

| Classe                      | F1-Score  | Precisão  | Revocação |
|-----------------------------|-----------|-----------|-----------|
| **Saúde**                    | 0.0       | 0.0       | 0.0       |
| **Ideologia**                | 0.6343    | 0.6728    | 0.6000    |
| **Insulto**                  | 0.8996    | 0.8578    | 0.9456    |
| **LGBTQIA+ Fobia**           | 0.4429    | 0.6739    | 0.3298    |
| **Outros Estilos de Vida**   | 0.2000    | 0.6667    | 0.1176    |
| **Aspectos Físicos**         | 0.3301    | 0.8500    | 0.2048    |
| **Profanidade/Obscenidade**  | 0.7730    | 0.7927    | 0.7543    |
| **Racismo**                  | 0.0       | 0.0       | 0.0       |
| **Intolerância Religiosa**   | 0.2893    | 0.5227    | 0.2000    |
| **Sexismo**                  | 0.0       | 0.0       | 0.0       |

---

- **Classe "Saúde"** e **"Racismo"** não apresentaram nenhum desempenho, com F1-Score, Precisão e Revocação iguais a 0. Isso pode indicar que o modelo não foi capaz de identificar exemplos dessas classes no conjunto de teste.
  
- **Classe "Insulto"** apresentou resultados muito bons, com F1-Score de 0.8996, indicando um bom equilíbrio entre Precisão (85.78%) e Revocação (94.56%).

- **Classe "LGBTQIA+ Fobia"** teve um desempenho menor, com F1-Score de 0.4429, refletindo uma menor capacidade de identificar corretamente os exemplos dessa categoria.

- **Classe "Outros Estilos de Vida"** também teve uma pontuação baixa de F1-Score (0.2000), com uma alta Precisão (66.67%) mas baixa Revocação (11.76%).

- **Classe "Aspectos Físicos"** obteve uma Precisão muito boa (85.00%), mas o F1-Score e a Revocação indicam que o modelo teve dificuldades em identificar corretamente os exemplos dessa classe.

- **Classe "Profanidade/Obscenidade"** obteve bons resultados com F1-Score de 0.7730, refletindo boa Precisão (79.27%) e Revocação (75.43%).

- **Classe "Intolerância Religiosa"** teve um desempenho baixo com F1-Score de 0.2893, devido à baixa Revocação (20.00%).

- **Classe "Sexismo"** também não apresentou desempenho, com F1-Score, Precisão e Revocação iguais a 0.

---

Esses resultados indicam áreas de sucesso e de melhoria para o modelo, sendo necessário um ajuste adicional para classes com desempenho insatisfatório, como "Saúde", "Racismo" e "Sexismo". A classe "Insulto" foi a mais bem identificada, e outras classes como "Profanidade/Obscenidade" e "Ideologia" apresentam bons resultados, mas ainda podem ser melhoradas.

Devido à maior importância da classe "Profanidade/Obscenidade", essencial para identificar linguagem imprópria, foram aplicadas medidas específicas para melhorar sua avaliação durante o treinamento. Essas medidas incluíram o uso de um threshold (limiar) de 70% e a aplicação da técnica de Class Weights, com um fator de 3. Isso significou atribuir maior peso a essa classe em relação às outras, para que o modelo focasse mais em identificar corretamente os exemplos de profanidade/obscenidade.

Com essa abordagem, foi possível aumentar a acurácia do modelo em 6%, alcançando o melhor desempenho na categoria de profanidade.

O recall, especificamente, foi beneficiado por essa estratégia. O recall é uma métrica que mede a capacidade do modelo de identificar corretamente todos os exemplos positivos de uma classe (neste caso, a classe "Profanidade/Obscenidade"). Uma maior ênfase nessa classe, por meio dos ajustes de threshold e Class Weights, permitiu uma maior recuperação de exemplos reais de profanidade, garantindo que o modelo identificasse de forma mais eficaz as instâncias relevantes dessa categoria. Isso foi crucial para melhorar o desempenho do modelo, especialmente em tarefas de moderação automática, onde a precisão na detecção de linguagem inadequada é fundamental.


####
Apos a conclusão suba o modelo para o huggie face e use ele nos testes
## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes. (Assumindo licença MIT com base no README anterior)