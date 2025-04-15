"""
Script para testar o modelo de análise de sentimento treinado.

Este script carrega o modelo treinado e permite fazer previsões de sentimento
para textos em português.
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_path: str = "ModeloSentimento"):
        """
        Inicializa o analisador de sentimento.
        
        Args:
            model_path: Caminho para o modelo treinado
        """
        logger.info(f"Carregando modelo de {model_path}...")
        
        # Carrega o modelo e o tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained("GargulaCapixava/ModeloLexiconPT")
        self.tokenizer = AutoTokenizer.from_pretrained("GargulaCapixava/ModeloLexiconPT")
        
        # Mapeamento de índices para sentimentos
        self.sentiment_map = {
            0: "Positivo",
            1: "Negativo",
            2: "Neutro"
        }
        
        logger.info("Modelo carregado com sucesso!")

    def predict(self, text: str) -> dict:
        """
        Faz a previsão de sentimento para um texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Dict com a previsão e as probabilidades
        """
        # Tokeniza o texto
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Faz a previsão
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
            probabilities = predictions[0].tolist()
        
        # Formata o resultado
        result = {
            "texto": text,
            "sentimento": self.sentiment_map[predicted_label],
            "probabilidades": {
                self.sentiment_map[i]: prob for i, prob in enumerate(probabilities)
            }
        }
        
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Faz previsões de sentimento para uma lista de textos.
        
        Args:
            texts: Lista de textos para análise
            
        Returns:
            Lista de dicionários com as previsões e probabilidades
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def main():
    """Função principal."""
    # Lista predefinida de textos para análise
    textos = [
        "Se mata",
        "Ansiedade já me dominou",
   
    ]
    
    # Inicializa o analisador
    analyzer = SentimentAnalyzer()
    
    # Faz as previsões
    results = analyzer.predict_batch(textos)
    
    # Exibe os resultados
    print("\nResultados da análise de sentimento:")
    print("-" * 50)
    for result in results:
        print(f"\nTexto: {result['texto']}")
        print(f"Sentimento: {result['sentimento']}")
        print("Probabilidades:")
        for sentiment, prob in result['probabilidades'].items():
            print(f"  {sentiment}: {prob:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    main() 