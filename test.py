# test.py

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from nltk.translate.bleu_score import sentence_bleu

def evaluate_caption(image_path, caption):
    """
    Evaluate the relevance of a caption for a given image using CLIP.
    Args:
        image_path (str): Path to the image file.
        caption (str): Caption to evaluate.
    Returns:
        float: Cosine similarity score between the image and caption embeddings.
    """
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image and text
    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True)

    # Get CLIP embeddings
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = torch.matmul(image_embeds, text_embeds.T).item()
    return similarity

def calculate_bleu_score(reference_caption, generated_caption):
    """
    Calculate the BLEU score for a generated caption against a reference caption.
    Args:
        reference_caption (str): The reference caption.
        generated_caption (str): The generated caption to evaluate.
    Returns:
        float: BLEU score.
    """
    reference = [reference_caption.split()]
    candidate = generated_caption.split()
    score = sentence_bleu(reference, candidate)
    return score

if __name__ == "__main__":
    # Example usage
    image_path = "images/example.jpg"  # Replace with an actual image path from the 'images' folder
    generated_caption = "A breathtaking view of the mountains under a clear blue sky."  # Replace with your generated caption
    reference_caption = "The mountains under a clear blue sky."  # Replace with a reference caption

    # Cosine similarity evaluation
    similarity_score = evaluate_caption(image_path, generated_caption)
    print(f"Cosine similarity score: {similarity_score}")

    # Provide an analysis based on the similarity score
    if similarity_score >= 0.7:
        print("Analysis: The caption is highly relevant to the image.")
    elif 0.4 <= similarity_score < 0.7:
        print("Analysis: The caption is moderately relevant to the image. Consider minor adjustments.")
    else:
        print("Analysis: The caption has low relevance to the image. Consider revising it.")

    # BLEU score evaluation
    bleu_score = calculate_bleu_score(reference_caption, generated_caption)
    print(f"BLEU score: {bleu_score}")

    # Provide an analysis based on the BLEU score
    if bleu_score >= 0.7:
        print("Analysis: The generated caption closely matches the reference caption.")
    elif 0.4 <= bleu_score < 0.7:
        print("Analysis: The generated caption is somewhat similar to the reference caption. Consider refinements.")
    else:
        print("Analysis: The generated caption differs significantly from the reference caption. Consider revising it.")
