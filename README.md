# Modelos de Classificação de Texto em Português com BERTimbau

Este repositório contém scripts para treinar e avaliar vários modelos de classificação de texto em português baseados na arquitetura BERTimbau. O foco está em tarefas como Análise de Sentimento, Detecção de Discurso de Ódio e, potencialmente, outras tarefas de PLN relacionadas usando datasets específicos em português.

## 📑 Sobre o Projeto

Este projeto fornece um framework e implementações específicas para o fine-tuning do modelo `neuralmind/bert-base-portuguese-cased` (BERTimbau) para tarefas de classificação de texto relevantes para a língua portuguesa. Inclui scripts para treinar modelos em datasets como LexiconPT (para Análise de Sentimento), HateBR e OLID-BR (provavelmente para Detecção de Discurso de Ódio/Linguagem Ofensiva).

## 🛠️ Tecnologias Utilizadas

*   **Python 3.x**
*   **PyTorch**: Framework de deep learning.
*   **Hugging Face Transformers**: Biblioteca para modelos de PLN de última geração, usada aqui para o BERTimbau.
*   **Hugging Face Datasets**: Biblioteca para carregar e processar datasets.
*   **Scikit-learn**: Para métricas de avaliação e funções utilitárias (como pesos de classe).
*   **Pandas**: Para manipulação de dados.
*   **NumPy**: Para operações numéricas.

## 📁 Estrutura do Projeto

```
.
├── Data-set/                 # Diretório contendo os datasets
│   ├── HateBR-main/          # Arquivos do dataset HateBR
│   ├── lexiconPT-master/     # Arquivos do dataset LexiconPT
│   └── ...                   # Outros datasets (ex: OLID-BR)
├── train_sentiment.py        # Script para treinar modelo de análise de sentimento (LexiconPT)
├── train_hatbr.py            # Script para treinar modelo no dataset HateBR
├── train_olid.py             # Script para treinar modelo no dataset OLID-BR
├── train_model.py            # Script genérico para fine-tuning do BERTimbau (potencialmente menos usado agora)
├── test_sentimento.py        # Script para testar o modelo de sentimento
├── test_hatbr.py             # Script para testar o modelo HateBR
├── test_OlydBr.py            # Script para testar o modelo OLID-BR
├── README.md                 # Este arquivo
└── ...                       # Outros arquivos de configuração/utilitários
```
*(Nota: A estrutura do dataset pode variar)*

## 🚀 Como Usar

### 1. Configuração

a.  **Clone o repositório:**
    ```bash
    git clone <url-do-repositorio>
    cd <diretorio-do-repositorio>
    ```

b.  **Instale as dependências:** É recomendado usar um ambiente virtual.
    ```bash
    pip install torch transformers datasets pandas numpy scikit-learn
    ```
    *(Você pode precisar de versões específicas com base na compatibilidade. Verifique os imports nos scripts se ocorrerem erros.)*

c.  **Baixe os Datasets:** Garanta que os datasets necessários (LexiconPT, HateBR, OLID-BR) sejam baixados e colocados corretamente na estrutura do diretório `Data-set/` conforme esperado pelos scripts de treinamento (ex: `Data-set/lexiconPT-master/lexiconPT-master/data/csv/oplexicon_v3.0.csv`). Verifique as funções `load_*_dataset` dentro de cada script `train_*.py` para os caminhos exatos esperados.

### 2. Treinando Modelos

Execute o script de treinamento específico para a tarefa desejada. Por exemplo, para treinar o modelo de análise de sentimento:

```bash
python train_sentiment.py --output_dir ./models/sentiment --epochs 3 --batch_size 8 --learning_rate 2e-5
```

Da mesma forma, execute `train_hatbr.py` ou `train_olid.py` para outras tarefas, ajustando os parâmetros conforme necessário:

```bash
python train_hatbr.py --output_dir ./models/hatbr --epochs 5 --batch_size 16
# Ajuste os parâmetros com base na configuração do argparse do script
```

*   Verifique a seção `argparse` dentro de cada script (`train_*.py`) para os argumentos de linha de comando disponíveis (como `--output_dir`, `--epochs`, `--batch_size`, `--learning_rate`, etc.).
*   O treinamento salvará o modelo fine-tuned e o tokenizador no diretório de saída especificado.

### 3. Avaliando Modelos

Execute o script de teste correspondente (ex: `test_sentimento.py`, `test_hatbr.py`) para avaliar um modelo treinado. Você provavelmente precisará indicar ao script de teste o diretório onde o modelo treinado foi salvo.

*(Exemplo - assumindo que o script de teste recebe o caminho do modelo como argumento)*
```bash
python test_sentimento.py --model_path ./models/sentiment
```
*   Verifique a configuração do `argparse` nos scripts `test_*.py` para uso específico.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes. (Assumindo licença MIT com base no README anterior)