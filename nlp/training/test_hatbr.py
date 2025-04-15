"""
Script para testar o modelo HateBR treinado.

Este script carrega o modelo treinado e avalia sua performance em exemplos de texto,
classificando-os em três categorias:
0: Não tóxico
1: Tóxico
2: Discurso de ódio
"""

import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

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

def load_model(model_path: str) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Carrega o modelo e tokenizer treinados.
    
    Args:
        model_path: Caminho para o diretório do modelo
        
    Returns:
        Tupla contendo o modelo e tokenizer
    """
    logger.info(f"Carregando modelo de {model_path}")
    
    # Carrega o modelo e tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def predict_text(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    max_length: int = 128
) -> Dict[str, float]:
    """
    Faz a predição para um texto.
    
    Args:
        model: Modelo carregado
        tokenizer: Tokenizer carregado
        text: Texto para classificar
        max_length: Tamanho máximo da sequência
        
    Returns:
        Dict com as probabilidades para cada classe
    """
    # Tokeniza o texto
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Faz a predição
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
    
    # Converte para dict com as classes
    probs = probabilities[0].numpy()
    return {LABEL_MAP[i]: float(prob) for i, prob in enumerate(probs)}

def evaluate_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    test_file: str,
    max_length: int = 128
) -> None:
    """
    Avalia o modelo em um conjunto de teste.
    
    Args:
        model: Modelo carregado
        tokenizer: Tokenizer carregado
        test_file: Caminho para o arquivo de teste
        max_length: Tamanho máximo da sequência
    """
    logger.info(f"Carregando arquivo de teste: {test_file}")
    
    # Carrega o arquivo de teste
    df = pd.read_csv(test_file)
    
    # Faz as predições
    predictions = []
    true_labels = []
    
    for _, row in df.iterrows():
        text = row['comentario']
        true_label = row['label_final']
        
        # Faz a predição
        probs = predict_text(model, tokenizer, text, max_length)
        pred_label = max(probs.items(), key=lambda x: x[1])[0]
        
        # Converte o nome da classe para o índice
        pred_idx = [k for k, v in LABEL_MAP.items() if v == pred_label][0]
        
        predictions.append(pred_idx)
        true_labels.append(true_label)
    
    # Calcula e exibe as métricas
    print("\nRelatório de Classificação:")
    print(classification_report(true_labels, predictions, target_names=list(LABEL_MAP.values())))
    
    # Cria e exibe a matriz de confusão
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(LABEL_MAP.values()),
                yticklabels=list(LABEL_MAP.values()))
    plt.title('Matriz de Confusão')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Predito')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
# Setup de logger (manter igual)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Substitua pelo nome correto do seu modelo no Hugging Face
MODEL_NAME = "GargulaCapixava/ModeloHateBR"

def load_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def main():
    # Carrega o modelo e tokenizer do Hugging Face
    try:
        model, tokenizer = load_model(MODEL_NAME)
        logger.info("Modelo carregado com sucesso da Hugging Face!")
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {str(e)}")
        return

    # Exemplos de teste
    test_texts = [
        "Que legal! Adorei o vídeo, muito informativo!",
        "Você é muito burro, não entende nada!",
        "Vai se foder, seu filho da puta!",
        "Esse povo é tudo uns vagabundos que não querem trabalhar!",
        "Parabéns pelo trabalho, ficou muito bom!",
        "Seu lugar é na cozinha, mulher!",
        "Volte para seu país, imigrante!",
        "Que vídeo legal, gostei muito!",
        "Você é muito incompetente, não serve para nada!",
        "Seu lugar é na cadeia, criminoso!"
    ]

    print("\nTestando exemplos individuais:")
    for text in test_texts:
        probs = predict_text(model, tokenizer, text)
        pred_label = max(probs.items(), key=lambda x: x[1])[0]
        print(f"\nTexto: {text}")
        print(f"Classificação: {pred_label}")
        print("Probabilidades:")
        for label, prob in probs.items():
            print(f"  {label}: {prob:.2%}")

    # Se ainda quiser avaliar com CSV, pode manter esta parte:
    # test_file = "app/nlp/Data-set/HateBR-main/HateBR-main/dataset/HateBR.csv"
    # evaluate_model(model, tokenizer, test_file)

if __name__ == "__main__":
    main()