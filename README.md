# Multimodal Sentiment Analysis using Transformers

## Overview
This project implements a **Multimodal Sentiment Analysis** system using **Transformers** for analyzing **Internet memes**. The model processes both **textual and visual data** to classify sentiment, humor, sarcasm, offensiveness, and motivational content. The project is designed to work with the **Memotion dataset** and is optimized for execution in **Google Colab**.

## Features
- **Sentiment Classification**: Classifies memes as **Positive (+1), Neutral (0), or Negative (-1)**.
- **Humor Classification**: Detects humor, sarcasm, and offensiveness as binary labels.
- **Semantic Scaling**: Predicts the degree (0-3 scale) of **Humor, Sarcasm, Offense, and Motivation**.
- **Multimodal Fusion**: Combines embeddings from **BERT** (text) and **ViT** (images) for robust classification.
- **Google Colab Ready**: Optimized for **GPU acceleration** and easy execution in Colab.

## Model Architecture
- **Text Processing**: BERT-based Transformer (`bert-base-uncased`) fine-tuned for meme text.
- **Image Processing**: Vision Transformer (`google/vit-base-patch16-224-in21k`) fine-tuned for meme images.
- **Fusion Layer**: Concatenates text and image embeddings.
- **Classification & Regression Heads**:
  - **Sentiment Classification**: (CrossEntropy Loss)
  - **Humor, Sarcasm, Offense, Motivation Classification**: (Binary classification, CrossEntropy Loss)
  - **Semantic Scaling (0-3 range)**: (Regression, MSE Loss)

## Dataset
- **Dataset Used**: Memotion Dataset (`Multimodal_Sentiment_Analysis_FinalAssignment.csv`)
- **Data Components**:
  - **Text**: Meme caption or embedded text.
  - **Image**: Meme image file.
  - **Labels**: Sentiment, Humor, Sarcasm, Offensiveness, Motivation.

## Installation
To run the code in **Google Colab**, execute the following steps:
```bash
# Install required libraries
pip install transformers torch torchvision pandas pillow scikit-learn
```

## Usage
### 1. Load Dataset
```python
csv_path = "/content/Multimodal_Sentiment_Analysis_FinalAssignment.csv"
image_folder = "/content/images"
dataset = MemotionDataset(csv_path, image_folder, tokenizer, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### 2. Train the Model
```python
train_model(model, dataloader, sentiment_criterion, regression_criterion, optimizer)
```

### 3. Save and Load Model
```python
torch.save(model.state_dict(), "multimodal_sentiment_model.pth")
model.load_state_dict(torch.load("multimodal_sentiment_model.pth"))
```

## Results & Evaluation
- **Evaluation Metrics**: Macro F1 Score for classification tasks.
- **Loss Functions**:
  - **CrossEntropyLoss**: Sentiment & Humor classification.
  - **MSELoss**: Semantic scale regression.
- The model outputs **classification predictions and continuous scale values** for memes.

## Future Work
- Implement **cross-attention mechanisms** for better fusion of text and image embeddings.
- Explore **larger multimodal transformers** like **CLIP** for improved performance.

## Contributors
- **Project Developed by**: Rohan Aditya
- **Based on the Memotion Dataset** and Transformer models.

## License
This project is licensed under the MIT License.

