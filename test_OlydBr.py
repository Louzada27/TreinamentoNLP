import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os
from transformers import AutoModel, AutoTokenizer


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Classes de toxicidade
LABEL_COLUMNS = [
    'health', 'ideology', 'insult', 'lgbtqphobia', 'other_lifestyle',
    'physical_aspects', 'profanity_obscene', 'racism', 'religious_intolerance', 'sexism'
]

# Tipos de insulto
INSULT_TYPES = {
    'TIN': 'Insulto direcionado',
    'UNT': 'Insulto não direcionado'
}

class ToxicityPredictor:
    def __init__(self):
        """
        Inicializa o preditor de toxicidade.
        
        Args:
            model_path: Caminho para o modelo treinado
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
     
        self.model = AutoModelForSequenceClassification.from_pretrained("")
        self.tokenizer = AutoTokenizer.from_pretrained("")

        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Modelo e tokenizer carregados com sucesso")
    
    def classify_insult_type(self, text: str) -> str:
        """
        Classifica o tipo de insulto (direcionado ou não direcionado).
        
        Args:
            text: Texto para analisar
            
        Returns:
            Tipo de insulto (TIN ou UNT)
        """
        # Palavras que indicam direcionamento
        direction_words = ['você', 'seu', 'sua', 'teu', 'tua', 'vocês', 'seus', 'suas', 'teus', 'tuas']
        
        # Verifica se o texto contém palavras de direcionamento
        text_lower = text.lower()
        has_direction = any(word in text_lower for word in direction_words)
        
        return 'TIN' if has_direction else 'UNT'
    
    def predict(self, text: str, threshold: float = 0.5) -> Tuple[List[str], List[float], str]:
        """
        Faz previsão de toxicidade para um texto.
        
        Args:
            text: Texto para analisar
            threshold: Limiar para considerar uma classe como positiva
            
        Returns:
            Tuple com lista de classes tóxicas, suas probabilidades e tipo de insulto
        """
        # Tokeniza o texto
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Faz a previsão
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)
        
        # Converte para numpy e pega as probabilidades
        probs = probabilities[0].cpu().numpy()
        
        # Identifica as classes tóxicas
        toxic_classes = []
        toxic_probs = []
        
        # Retorna todas as classes e suas probabilidades
        for i, prob in enumerate(probs):
            toxic_classes.append(LABEL_COLUMNS[i])
            toxic_probs.append(float(prob))
        
        # Classifica o tipo de insulto se houver insulto
        insult_type = None
        if 'insult' in toxic_classes:
            insult_type = self.classify_insult_type(text)
        
        return toxic_classes, toxic_probs, insult_type
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Tuple[List[str], List[float], str]]:
        """
        Faz previsão de toxicidade para uma lista de textos.
        
        Args:
            texts: Lista de textos para analisar
            threshold: Limiar para considerar uma classe como positiva
            
        Returns:
            Lista de tuplas com classes tóxicas, suas probabilidades e tipo de insulto
        """
        # Tokeniza os textos
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Faz a previsão
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)
        
        # Converte para numpy e pega as probabilidades
        probs = probabilities.cpu().numpy()
        
        # Processa cada texto
        results = []
        for text, text_probs in zip(texts, probs):
            toxic_classes = []
            toxic_probs = []
            
            # Retorna todas as classes e suas probabilidades
            for i, prob in enumerate(text_probs):
                toxic_classes.append(LABEL_COLUMNS[i])
                toxic_probs.append(float(prob))
            
            # Classifica o tipo de insulto se houver insulto
            insult_type = None
            if 'insult' in toxic_classes:
                insult_type = self.classify_insult_type(text)
            
            results.append((toxic_classes, toxic_probs, insult_type))
        
        return results

def print_results(test_texts: List[str], results: List[Tuple[List[str], List[float], Optional[int]]]) -> None:
    """
    Exibe os resultados da classificação de forma formatada.
    
    Args:
        test_texts: Lista de textos de teste
        results: Lista de tuplas (classes, probabilidades, tipo_insulto)
    """
    for text, (classes, probs, insult_type) in zip(test_texts, results):
        logger.info(f"\nTexto: {text}")
        logger.info("Probabilidades de toxicidade:")
        for cls, prob in zip(classes, probs):
            logger.info(f"- {cls}: {prob:.2%}")
            if cls == 'insult' and insult_type:
                logger.info(f"  Tipo de insulto: {INSULT_TYPES[insult_type]} ({insult_type})")
        logger.info("-" * 50)

def main():
    # Exemplos de teste
    test_texts = [
        # Exemplos de insultos direcionados (TIN)
      
        "Caralho, você e um bixa do caralho"
   
    ]
    
    # Inicializa o preditor
    predictor = ToxicityPredictor()
    
    # Faz previsões
    logger.info("Iniciando previsões...")
    results = predictor.predict_batch(test_texts)
    
    # Mostra resultados
    logger.info("\nResultados das previsões:")
    logger.info("-" * 50)
    
    print_results(test_texts, results)

if __name__ == "__main__":
    main() 


