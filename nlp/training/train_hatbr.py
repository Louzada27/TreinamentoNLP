"""
Script para fine-tuning de modelos BERTimbau para detecção de toxicidade usando o dataset HateBR.

Este script fornece funções otimizadas para treinar modelos de detecção de toxicidade
usando o dataset HateBR, que contém comentários tóxicos em português classificados em 3 níveis:
0: Não tóxico
1: Tóxico
2: Discurso de ódio
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
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
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Mapeamento de rótulos para nomes descritivos
LABEL_MAP = {
    0: "não tóxico",
    1: "tóxico",
    2: "discurso de ódio"
}

def load_hatebr_dataset() -> Dict[str, Any]:
    """
    Carrega o dataset HateBR do arquivo local.
    
    Returns:
        Dict contendo os datasets de treino e teste
    """
    logger.info("Carregando dataset HateBR...")
    
    # Caminho para o arquivo CSV
    dataset_path = os.path.join("app", "nlp", "Data-set", "HateBR-main", "HateBR-main", "dataset", "HateBR.csv")
    
    # Carrega o dataset
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset carregado com {len(df)} exemplos")
    
    # Verifica se as colunas necessárias existem
    required_columns = ['comentario', 'label_final']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas necessárias ausentes no dataset: {missing_columns}")
    
    # Verifica se os rótulos estão no formato correto (0, 1, 2)
    unique_labels = df['label_final'].unique()
    if not all(label in [0, 1, 2] for label in unique_labels):
        raise ValueError(f"Rótulos inválidos encontrados: {unique_labels}. Esperado: [0, 1, 2]")
    
    # Divide em treino e teste (80% treino, 20% teste)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Converte para o formato datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    logger.info(f"Dataset dividido - Treino: {len(train_dataset)} exemplos, Teste: {len(test_dataset)} exemplos")
    
    # Log da distribuição dos rótulos
    for split_name, split_df in [("Treino", train_df), ("Teste", test_df)]:
        label_counts = split_df['label_final'].value_counts().sort_index()
        logger.info(f"\nDistribuição de rótulos no conjunto de {split_name}:")
        for label, count in label_counts.items():
            logger.info(f"{LABEL_MAP[label]}: {count} exemplos ({count/len(split_df)*100:.1f}%)")
    
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
        dataset: Dataset HateBR
        tokenizer: Tokenizador BERT
        max_length: Tamanho máximo da sequência
        
    Returns:
        Datasets processados
    """
    logger.info(f"Pré-processando datasets com max_length={max_length}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['comentario'],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Processa cada split do dataset separadamente
    processed_datasets = {}
    for split_name, split_dataset in dataset.items():
        logger.info(f"Processando split: {split_name}")
        
        # Tokeniza o dataset
        tokenized_dataset = split_dataset.map(tokenize_function, batched=True)
        
        # Renomeia a coluna de rótulos
        def rename_labels(examples):
            return {'labels': examples['label_final']}
        
        # Aplica renomeação de rótulos
        tokenized_dataset = tokenized_dataset.map(rename_labels, batched=False)
        
        # Remove colunas originais
        tokenized_dataset = tokenized_dataset.remove_columns(
            ['comentario', 'label_final', 'id', 'anotator1', 'anotator2', 'anotator3', 'links_post', 'account_post']
        )
        
        # Configura formato para treinamento
        tokenized_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
        )
        
        processed_datasets[split_name] = tokenized_dataset
    
    return processed_datasets

def compute_metrics(pred):
    """
    Calcula métricas de avaliação para classificação multi-classe.
    
    Args:
        pred: Predições do modelo
        
    Returns:
        Dict com as métricas calculadas
    """
    # Converte tensores para arrays NumPy
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Métricas gerais
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    # Relatório detalhado por classe
    report = classification_report(
        labels, 
        preds, 
        target_names=list(LABEL_MAP.values()),
        labels=list(LABEL_MAP.keys()),  # Força o uso de todas as classes
        output_dict=True
    )
    
    # Métricas por classe
    per_class_metrics = {}
    for label, name in LABEL_MAP.items():
        per_class_metrics[name] = {
            'precision': float(report[name]['precision']),
            'recall': float(report[name]['recall']),
            'f1': float(report[name]['f1-score'])
        }
    
    return {
        # Métricas gerais
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        
        # Métricas por classe
        'per_class_metrics': per_class_metrics
    }

def train_model(
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    output_dir: str = "./results",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 128
) -> BertForSequenceClassification:
    """
    Treina um modelo BERTimbau para detecção de toxicidade.
    
    Args:
        model_name: Nome do modelo base
        output_dir: Diretório para salvar o modelo
        epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        max_length: Tamanho máximo da sequência
        
    Returns:
        Modelo treinado
    """
    logger.info(f"Iniciando treinamento do modelo {model_name}")
    
    # Carrega o dataset
    dataset = load_hatebr_dataset()
    
    # Inicializa o tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Pré-processa os dados
    tokenized_datasets = preprocess_data(dataset, tokenizer, max_length)
    
    # Carrega o modelo para classificação de 3 classes
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
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
        # Configurações otimizadas para CPU
        use_cpu=True,  # Usa CPU em vez de no_cuda
        fp16=False,  # Desabilita precisão mista
        gradient_accumulation_steps=4,  # Acumular gradientes para simular batch maior
        report_to="tensorboard",  # Usar TensorBoard para visualização
        remove_unused_columns=True,  # Remover colunas não utilizadas automaticamente
    )
    
    # Configura o trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Treina o modelo
    logger.info("Iniciando treinamento...")
    trainer.train()
    
    # Salva o modelo e tokenizer
    logger.info(f"Salvando modelo treinado em {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Treinamento de modelo BERTimbau para detecção de toxicidade')
    
    parser.add_argument('--model_name', type=str, default="neuralmind/bert-base-portuguese-cased",
                        help='Nome do modelo base')
    parser.add_argument('--output_dir', type=str, default="./results",
                        help='Diretório para salvar o modelo')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Tamanho do batch')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Taxa de aprendizado')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Tamanho máximo da sequência')
    
    args = parser.parse_args()
    
    # Treina o modelo
    model = train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    logger.info("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main() 