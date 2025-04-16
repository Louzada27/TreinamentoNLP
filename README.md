# 🇧🇷 Modelos de Classificação de Texto em Português com BERTimbau

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

. ├── Data-set/ # Diretório contendo os datasets │ ├── HateBR-main/ # Arquivos do dataset HateBR │ ├── lexiconPT-master/ # Arquivos do dataset LexiconPT │ └── ...
├── train_sentiment.py # Script para treinar modelo de análise de sentimento (LexiconPT) ├── train_hatbr.py # Script para treinar modelo no dataset HateBR ├── train_olid.py # Script para treinar modelo no dataset OLID-BR ├── train_model.py # Script genérico para fine-tuning do BERTimbau (potencialmente menos usado agora) ├── test_sentimento.py # Script para testar o modelo de sentimento ├── test_hatbr.py # Script para testar o modelo HateBR ├── test_OlydBr.py # Script para testar o modelo OLID-BR ├── README.md # Este arquivo └── ... # Outros arquivos de configuração/utilitários
*(Nota: A estrutura do dataset pode variar)*

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

🚀 **Como usar:**

Execute o script com os parâmetros desejados:

```bash
python train_sentiment.py --epochs 4 --batch_size 16 --learning_rate 3e-5
OLID-BR
🧠 Fine-Tuning BERTimbau para Detecção de Toxicidade com OLID-BR

Este projeto realiza o fine-tuning de modelos BERT em português (como o BERTimbau) para classificação multilabel de toxicidade utilizando o dataset OLID-BR, uma versão brasileira do OLID (Offensive Language Identification Dataset).

Funcionalidades:

Treinamento de modelos BERT para detecção de múltiplas categorias de toxicidade

Cálculo de métricas detalhadas (macro, micro, por classe, etc.)

Aplicação de thresholds específicos por classe

Salvamento de logs e métricas de avaliação

Suporte a pesos customizados (pos_weight) para lidar com desbalanceamento

3. Avaliando Modelos
Modelo de Análise de Sentimento (BERTimbau + LexiconPT)
Este modelo foi treinado para classificar sentimentos em três categorias: Positivo, Negativo e Neutro. A seguir, os principais resultados obtidos durante a avaliação no conjunto de teste:

Métricas Gerais:

Acurácia: 76.84%

Precisão: 76.75%

Revocação: 76.84%

F1-Score: 76.63%

Matriz de Confusão:


Predito: Positivo	Predito: Negativo	Predito: Neutro
Real: Positivo	1052	292
Real: Negativo	118	2586
Real: Neutro	272	195
Desempenho por Classe:

Positivo:

Precisão: 72.95%

Revocação: 61.74%

F1-Score: 66.88%

Negativo:

Precisão: 84.15%

Revocação: 87.42%

F1-Score: 85.76%

Neutro:

Precisão: 68.07%

Revocação: 73.70%

F1-Score: 70.78%

Modelo de Classificação de Toxicidade (ModeloToxidade2.0)
Métricas Gerais:

Acurácia: 96.71%

Precisão: 96.72%

Revocação: 96.71%

F1-Score: 96.71%

Matriz de Confusão:


Predito: Não Tóxico	Predito: Tóxico
Real: Não Tóxico	3369
Real: Tóxico	99
Desempenho por Classe:

Não Tóxico:

Precisão: 97.15%

Revocação: 96.26%

F1-Score: 96.70%

Tóxico:

Precisão: 96.29%

Revocação: 97.17%

F1-Score: 96.73%

Detecção de Profanidade/Obscenidade
Métricas:

Precisão: 76.44%

Revocação: 78.03%

F1-Score: 77.23%

🚨 Importância: Esta etapa é crucial para a segurança online de públicos sensíveis (como crianças), sendo uma ferramenta eficaz para sistemas de moderação automática em ambientes virtuais.

4. Subindo o Modelo para o Hugging Face
Após a conclusão dos treinamentos, o modelo pode ser facilmente carregado e testado no Hugging Face para facilitar o uso em produção.

📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

Agora, a seção de resultados e explicações do projeto foi devidamente organizada e formatada para um arquivo `README.md` com a estrutura correta.
