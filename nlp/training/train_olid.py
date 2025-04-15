"""
Script para fine-tuning de modelos BERTimbau para detecção de toxicidade usando o dataset OLID-BR.

Este script fornece funções otimizadas para treinar modelos de detecção de toxicidade
usando o dataset OLID-BR, que contém múltiplas categorias de toxicidade.
"""

import os
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Definição das colunas de rótulos do OLID-BR
LABEL_COLUMNS = [
    'health', 'ideology', 'insult', 'lgbtqphobia', 'other_lifestyle', 
    'physical_aspects', 'profanity_obscene', 'racism', 'sexism', 'xenophobia'
]

# Definição dos thresholds por classe
THRESHOLDS = {
    'profanity_obscene': 0.7,
}

def load_olid_dataset() -> Dict[str, Any]:
    """
    Carrega o dataset OLID-BR do Hugging Face.
    
    Returns:
        Dict contendo os datasets de treino e teste
    """
    logger.info("Carregando dataset OLID-BR...")
    dataset = load_dataset("dougtrajano/olid-br")
    logger.info(f"Dataset carregado - Treino: {len(dataset['train'])} exemplos, Teste: {len(dataset['test'])} exemplos")
    
    # Verifica se todas as colunas de rótulos existem no dataset
    missing_columns = [col for col in LABEL_COLUMNS if col not in dataset['train'].features]
    if missing_columns:
        raise ValueError(f"Colunas de rótulos ausentes no dataset: {missing_columns}")
    
    return dataset

def preprocess_data(
    dataset: Dict[str, Any],
    tokenizer: BertTokenizer,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Pré-processa os datasets para treino.
    
    Args:
        dataset: Dataset OLID-BR
        tokenizer: Tokenizador BERT
        max_length: Tamanho máximo da sequência
        
    Returns:
        Datasets processados
    """
    logger.info(f"Pré-processando datasets com max_length={max_length}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokeniza os datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Prepara os rótulos
    all_labels = []
    for example in dataset['train']:
        labels = [label for label in LABEL_COLUMNS if example[label] == 1]
        all_labels.append(labels)
    
    # Ajusta o MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    
    def format_labels(examples):
        labels = [label for label in LABEL_COLUMNS if examples[label] == 1]
        binarized_labels = mlb.transform([labels])[0]
        return {'labels': binarized_labels}
    
    # Aplica formatação de rótulos
    tokenized_datasets = tokenized_datasets.map(format_labels, batched=False)
    
    # Remove colunas originais de rótulos
    tokenized_datasets = tokenized_datasets.remove_columns(LABEL_COLUMNS)
    
    # Configura formato para treinamento
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    
    return tokenized_datasets, mlb

def apply_thresholds(preds, thresholds, label_list):
    """
    Aplica thresholds específicos para cada classe nas predições.
    
    Args:
        preds: Array de probabilidades (batch_size, num_labels)
        thresholds: Dicionário com thresholds por classe
        label_list: Lista de labels na ordem correta
        
    Returns:
        Array binarizado com as predições após aplicar os thresholds
    """
    binarized = []
    for prob in preds:
        row = []
        for i, p in enumerate(prob):
            th = thresholds.get(label_list[i], 0.15)  # valor padrão
            row.append(1 if p >= th else 0)
        binarized.append(row)
    return np.array(binarized)

def compute_metrics(eval_pred):
    """
    Calcula métricas de avaliação para classificação multi-label.
    
    Args:
        eval_pred: Predições e rótulos reais do modelo
        
    Returns:
        Dict com as métricas calculadas
    """
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions))
    
    # Aplica thresholds específicos por classe
    predictions = apply_thresholds(predictions.numpy(), THRESHOLDS, LABEL_COLUMNS)
    
    # Calcula métricas para cada classe
    per_class_metrics = {}
    for i, label in enumerate(LABEL_COLUMNS):
        precision = precision_score(labels[:, i], predictions[:, i], average='binary', zero_division=0)
        recall = recall_score(labels[:, i], predictions[:, i], average='binary', zero_division=0)
        f1 = f1_score(labels[:, i], predictions[:, i], average='binary', zero_division=0)
        per_class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calcula métricas macro (média das métricas por classe)
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Calcula métricas micro (considerando todos os exemplos)
    micro_precision = precision_score(labels, predictions, average='micro', zero_division=0)
    micro_recall = recall_score(labels, predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    
    # Calcula Hamming Loss e Jaccard Score
    hamming_loss_value = hamming_loss(labels, predictions)
    jaccard_score_value = jaccard_score(labels, predictions, average='samples')
    
    # Calcula acurácia
    accuracy = 1 - hamming_loss_value
    
    return {
        'eval_accuracy': accuracy,
        'eval_precision': macro_precision,
        'eval_recall': macro_recall,
        'eval_f1': macro_f1,
        'eval_hamming_loss': hamming_loss_value,
        'eval_jaccard_score': jaccard_score_value,
        'eval_micro_precision': micro_precision,
        'eval_micro_recall': micro_recall,
        'eval_micro_f1': micro_f1,
        'eval_per_class_metrics': per_class_metrics
    }

class MultiLabelTrainer(Trainer):
    """Trainer personalizado para classificação multi-label."""
    
    def __init__(self, **kwargs):
        self.pos_weight = kwargs.pop('pos_weight', None)
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Calcula a perda para classificação multi-label.
        
        Args:
            model: Modelo a ser treinado
            inputs: Dicionário com as entradas do modelo
            return_outputs: Se True, retorna também as saídas do modelo
            num_items_in_batch: Número de itens no batch (ignorado)
            
        Returns:
            Perda calculada e opcionalmente as saídas do modelo
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(model.device))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

def train_model(
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    output_dir: str = "./results",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256
) -> Tuple[BertForSequenceClassification, MultiLabelBinarizer]:
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
        Tuple com o modelo treinado e o MultiLabelBinarizer
    """
    logger.info(f"Iniciando treinamento do modelo {model_name}")
    
    # Carrega o dataset
    dataset = load_olid_dataset()
    
    # Inicializa o tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Pré-processa os dados
    tokenized_datasets, mlb = preprocess_data(dataset, tokenizer, max_length)
    
    # Obtém o número correto de labels do MultiLabelBinarizer
    num_labels = len(mlb.classes_)
    logger.info(f"Número de labels no dataset: {num_labels}")
    logger.info(f"Classes no dataset: {mlb.classes_}")
    logger.info(f"Classes definidas: {LABEL_COLUMNS}")
    
    # Verifica se as classes correspondem
    if set(mlb.classes_) != set(LABEL_COLUMNS):
        raise ValueError(f"Classes no dataset ({set(mlb.classes_)}) não correspondem às classes definidas ({set(LABEL_COLUMNS)})")
    
    # Carrega o modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,  # Usa o número de labels do dataset
        problem_type="multi_label_classification"
    )
    
    # Define os pesos para cada classe de toxicidade
    pos_weight = torch.tensor([
        1.0,  # health
        1.0,  # ideology
        1.0,  # insult
        1.0,  # lgbtqphobia
        1.0,  # other_lifestyle
        1.0,  # physical_aspects
        3.0,  # profanity_obscene (peso 3x maior)
        1.0,  # racism
        1.0,  # sexism
        1.0   # xenophobia
    ])
    
    # Verifica se o número de pesos corresponde ao número de classes
    if len(pos_weight) != len(LABEL_COLUMNS):
        raise ValueError(f"Número de pesos ({len(pos_weight)}) não corresponde ao número de classes ({len(LABEL_COLUMNS)})")
    
    # Configura o treinamento
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
        greater_is_better=True
    )
    
    # Configura o treinador com os pesos
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # Usa o conjunto de teste para validação
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        pos_weight=pos_weight
    )
    
    # Treina o modelo
    logger.info("Iniciando treinamento...")
    trainer.train()
    
    # Salva o modelo e tokenizer
    logger.info(f"Salvando modelo treinado em {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, mlb

def save_evaluation_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Salva os resultados da avaliação em um arquivo de texto.
    
    Args:
        results: Dicionário com os resultados da avaliação
        output_dir: Diretório onde o arquivo será salvo
    """
    try:
        # Cria o diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Gera o nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.txt")
        
        # Abre o arquivo em modo de escrita
        with open(output_file, 'w', encoding='utf-8') as f:
            # Escreve o cabeçalho
            f.write("=" * 80 + "\n")
            f.write(f"Resultados da Avaliação - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Escreve as métricas gerais
            f.write("Métricas Gerais:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Acurácia: {results['eval_accuracy']:.4f}\n")
            f.write(f"Precisão: {results['eval_precision']:.4f}\n")
            f.write(f"Revocação: {results['eval_recall']:.4f}\n")
            f.write(f"F1-Score: {results['eval_f1']:.4f}\n")
            f.write(f"Hamming Loss: {results['eval_hamming_loss']:.4f}\n")
            f.write(f"Jaccard Score: {results['eval_jaccard_score']:.4f}\n")
            f.write(f"Precisão Micro: {results['eval_micro_precision']:.4f}\n")
            f.write(f"Revocação Micro: {results['eval_micro_recall']:.4f}\n")
            f.write(f"F1-Score Micro: {results['eval_micro_f1']:.4f}\n\n")
            
            # Escreve as métricas por classe
            f.write("Métricas por Tipo de Toxicidade:\n")
            f.write("-" * 40 + "\n")
            for label, metrics in results['eval_per_class_metrics'].items():
                f.write(f"\n{label.upper()}:\n")
                f.write(f"  Precisão: {metrics['precision']:.4f}\n")
                f.write(f"  Revocação: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
            
            # Escreve os thresholds utilizados
            f.write("\nThresholds Utilizados:\n")
            f.write("-" * 40 + "\n")
            for label, threshold in THRESHOLDS.items():
                f.write(f"{label}: {threshold:.2f}\n")
            
        logger.info(f"Resultados salvos em: {output_file}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar resultados: {str(e)}")
        raise

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
    model, mlb = train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    logger.info("Treinamento concluído com sucesso!")
    
    # Avalia o modelo no conjunto de teste
    dataset = load_olid_dataset()
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    tokenized_datasets, _ = preprocess_data(dataset, tokenizer, args.max_length)
    
    # Cria o trainer para avaliação
    trainer = MultiLabelTrainer(
        model=model,
        compute_metrics=compute_metrics
    )
    
    # Obtém as métricas
    metrics = trainer.evaluate(tokenized_datasets['test'])
    
    # Salva os resultados
    save_evaluation_results(
        results=metrics,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 