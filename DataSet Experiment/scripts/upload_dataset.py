"""Standalone script to upload dataset to Langfuse"""

import sys
sys.path.append('..')

from langfuse import get_client
from data.evaluation_dataset import EVALUATION_DATASET
from utils.utils import load_config, validate_langfuse_connection, logger

def main():
    config = load_config()
    langfuse = get_client()
    
    if not validate_langfuse_connection(langfuse):
        raise RuntimeError("❌ Langfuse authentication failed")
    
    DATASET_NAME = config['dataset']['name']
    
    # Create or get dataset
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
        logger.info(f"ℹ️ Dataset '{DATASET_NAME}' already exists.")
    except Exception:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="Multi-turn conversations for evaluation"
        )
        logger.info(f"✅ Created dataset '{DATASET_NAME}'.")
    
    # Upload items
    logger.info("⬆️ Uploading dataset items...")
    for i, item in enumerate(EVALUATION_DATASET):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["input"]
        )
        logger.info(f"   Uploaded item {i + 1}/{len(EVALUATION_DATASET)}")
    
    logger.info("\n🎉 Dataset upload completed successfully!")

if __name__ == "__main__":
    main()