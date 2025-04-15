"""
Script para converter arquivos RDA do LexiconPT para CSV.

Este script converte os arquivos RDA do dataset LexiconPT para o formato CSV,
facilitando o uso em outros contextos e frameworks.
"""

import os
import pandas as pd
import pyreadr
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def convert_rda_to_csv(
    input_dir: str,
    output_dir: str,
    files: List[str] = None
) -> None:
    """
    Converte arquivos RDA para CSV.
    
    Args:
        input_dir: Diretório contendo os arquivos RDA
        output_dir: Diretório onde os arquivos CSV serão salvos
        files: Lista de arquivos para converter. Se None, converte todos os arquivos RDA
    """
    # Cria o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Se nenhum arquivo específico foi fornecido, lista todos os arquivos RDA
    if files is None:
        files = [f for f in os.listdir(input_dir) if f.endswith('.rda')]
    
    logger.info(f"Encontrados {len(files)} arquivos RDA para converter")
    
    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace('.rda', '.csv'))
        
        logger.info(f"Convertendo {file}...")
        
        try:
            # Lê o arquivo RDA
            result = pyreadr.read_r(input_path)
            
            # O resultado é um dicionário onde a chave é o nome do objeto R
            # e o valor é o DataFrame
            df = list(result.values())[0]
            
            # Log das informações do dataset
            logger.info(f"Colunas disponíveis: {df.columns.tolist()}")
            logger.info(f"Número de linhas: {len(df)}")
            
            # Salva como CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Arquivo salvo em: {output_path}")
            
        except Exception as e:
            logger.error(f"Erro ao converter {file}: {str(e)}")

def main():
    """Função principal."""
    # Diretórios
    input_dir = os.path.join("app", "nlp", "Data-set", "lexiconPT-master", "lexiconPT-master", "data")
    output_dir = os.path.join("app", "nlp", "Data-set", "lexiconPT-master", "lexiconPT-master", "data", "csv")
    
    # Lista de arquivos específicos para converter (opcional)
    files_to_convert = [
        "oplexicon_v2.1.rda",
    ]
    
    # Converte os arquivos
    convert_rda_to_csv(input_dir, output_dir, files_to_convert)

if __name__ == "__main__":
    main()
