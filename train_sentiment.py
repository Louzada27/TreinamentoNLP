"""
Script para treinar o modelo BERTimbau para análise de sentimento usando o dataset LexiconPT.

Este script carrega o dataset LexiconPT, pré-processa os dados e treina um modelo BERTimbau
para classificação de sentimento em português.
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import logging
import argparse
from typing import Dict, List, Tuple, Any
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # Garante que os rótulos são inteiros
        self.labels = [int(label) if label is not None else 0 for label in labels]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_lexiconpt_dataset() -> Dict[str, Any]:
    """
    Carrega o dataset LexiconPT.
    
    Returns:
        Dict contendo os datasets de treino e teste
    """
    logger.info("Carregando dataset LexiconPT...")
    
    # Caminho para o arquivo CSV do dataset
    dataset_path = os.path.join("app", "nlp", "Data-set", "lexiconPT-master", "lexiconPT-master", "data", "csv", "oplexicon_v3.0.csv")
    
    # Carrega o dataset
    df = pd.read_csv(dataset_path)
    
    # Log das colunas disponíveis e valores únicos
    logger.info(f"Colunas disponíveis no dataset: {df.columns.tolist()}")
    logger.info(f"Valores únicos na coluna polarity: {df['polarity'].unique()}")
    logger.info(f"Número de exemplos antes do processamento: {len(df)}")
    
    # Mapeia os valores de polaridade para os rótulos numéricos
    polarity_map = {
        1: 0,    # Positivo
        -1: 1,   # Negativo
        0: 2     # Neutro
    }
    
    # Log dos valores antes do mapeamento
    logger.info("Distribuição de polaridade antes do mapeamento:")
    logger.info(df['polarity'].value_counts().to_dict())
    
    # Converte polaridade para valores numéricos
    df['polarity'] = df['polarity'].map(polarity_map)
    
    # Log dos valores após o mapeamento
    logger.info("Distribuição de polaridade após o mapeamento:")
    logger.info(df['polarity'].value_counts().to_dict())
    
    # Remove linhas com valores None na polaridade
    df = df.dropna(subset=['polarity'])
    
    # Log após remover valores None
    logger.info(f"Número de exemplos após remover valores None: {len(df)}")
    
    # Converte polaridade para inteiros
    df['polarity'] = df['polarity'].astype(int)
    
    logger.info(f"Dataset carregado com {len(df)} exemplos")
    logger.info(f"Distribuição final de sentimento: {df['polarity'].value_counts().to_dict()}")
    
    if len(df) == 0:
        raise ValueError("Dataset vazio após processamento. Verifique o mapeamento de polaridade.")
    
    # Divide em treino e teste (80% treino, 20% teste)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Converte para o formato datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    logger.info(f"Dataset dividido - Treino: {len(train_dataset)} exemplos, Teste: {len(test_dataset)} exemplos")
    
    return {
        'train': train_dataset,
        'test': test_dataset
    }

def preprocess_data(
    dataset: Dict[str, Any],
    tokenizer: BertTokenizer,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Pré-processa os datasets para treino.
    
    Args:
        dataset: Dicionário contendo os datasets de treino e teste
        tokenizer: Tokenizador BERT
        max_length: Tamanho máximo da sequência
        
    Returns:
        Dict contendo os datasets processados
    """
    logger.info(f"Pré-processando datasets com max_length={max_length}")
    
    processed_datasets = {}
    for split_name, split_dataset in dataset.items():
        logger.info(f"Processando dataset {split_name}...")
        
        # Obtém textos e rótulos
        text_column = 'term' if 'term' in split_dataset.column_names else 'word'
        texts = split_dataset[text_column]
        labels = split_dataset['polarity']
        
        # Tokeniza os textos
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Cria o dataset
        processed_datasets[split_name] = SentimentDataset(encodings, labels)
        
        logger.info(f"Dataset {split_name} processado com {len(processed_datasets[split_name])} exemplos")
    
    return processed_datasets

def compute_metrics(pred):
    """
    Calcula métricas de avaliação.
    
    Args:
        pred: Predições do modelo
        
    Returns:
        Dict com as métricas calculadas
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }

# Custom Trainer com suporte a pesos de classe
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(
             weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
         )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss






def train_model(
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    output_dir: str = "./results",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256
) -> None:
    

    """
    
    Treina o modelo de análise de sentimento.
    
    Args:
        model_name: Nome do modelo base
        output_dir: Diretório para salvar o modelo
        epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        max_length: Tamanho máximo da sequência
    """
    # Carrega o dataset
    dataset = load_lexiconpt_dataset()
    
    # Carrega o tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Pré-processa os dados
    tokenized_datasets = preprocess_data(dataset, tokenizer, max_length)
    
    # --- Calculate Class Weights ---
    train_labels = tokenized_datasets['train'].labels
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Calculated class weights: {class_weights_tensor}")
    # --- End Calculate Class Weights ---
    
    # Carrega o modelo
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # Positivo, Negativo, Neutro
        problem_type="single_label_classification"
    )
    
    # Configura os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=learning_rate,
        fp16=True,
        gradient_accumulation_steps=4,
        report_to="tensorboard",
        remove_unused_columns=False,  # Importante: não remover colunas não utilizadas
        label_names=["labels"],  # Especifica o nome da coluna de rótulos
    )
    
    # Cria o trainer (using WeightedTrainer)
    trainer = WeightedTrainer( # Changed from Trainer
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights_tensor # Pass weights here
    )
    
    # Treina o modelo
    logger.info("Iniciando treinamento...")
    trainer.train()
    
    # Salva o modelo
    logger.info("Salvando modelo...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Avalia o modelo
    logger.info("Avaliando modelo...")
    eval_results = trainer.evaluate()
    logger.info(f"Resultados da avaliação: {eval_results}")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Treina modelo de análise de sentimento")
    parser.add_argument("--model_name", type=str, default="neuralmind/bert-base-portuguese-cased",
                      help="Nome do modelo base")
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="Diretório para salvar o modelo")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Tamanho do batch")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Taxa de aprendizado")
    parser.add_argument("--max_length", type=int, default=128,
                      help="Tamanho máximo da sequência")
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main() 