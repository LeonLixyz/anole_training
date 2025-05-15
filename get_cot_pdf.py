import json
import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Paths
predictions_json_path = "/workspace/anole_training/outputs/vlm_reasoning_eval/vlm_reasoning_evalimage_seq_len-1024-anole-hyper-train1val1lr3e-05-geometry_reasoning-prompt_anole-42/predictions_eval_None.json"
output_pdf_path = "/workspace/anole_training/reasoning_traces.pdf"

# Load predictions data
with open(predictions_json_path, 'r') as f:
    predictions_data = json.load(f)

# Setup PDF document
doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
styles = getSampleStyleSheet()
content = []

# Create custom styles
header_style = ParagraphStyle(
    'Header',
    parent=styles['Heading1'],
    fontSize=14,
    textColor=colors.blue,
    spaceAfter=12
)

subheader_style = ParagraphStyle(
    'SubHeader',
    parent=styles['Heading2'],
    fontSize=12,
    textColor=colors.darkblue,
    spaceAfter=6
)

text_style = ParagraphStyle(
    'Text',
    parent=styles['Normal'],
    fontSize=10,
    spaceAfter=10
)

def process_text_and_images(text, img_paths, max_width=5*inch, max_height=4*inch):
    """Process text with <image> tags and replace with actual images"""
    elements = []
    
    # Handle case where img_paths is None
    if img_paths is None:
        img_paths = []
    
    # Handle special image tags format
    text = re.sub(r'<image_start>\[reasoning_image_\d+\]<image_end>', "<image>", text)
    
    # Split by <image> tag
    parts = text.split("<image>")
    
    # Add first text part
    if parts[0]:
        elements.append(Paragraph(parts[0], text_style))
    
    # Add images and remaining text parts
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        if i < len(parts) - 1:  # Make sure we have a text part after this image
            try:
                if os.path.exists(img_path):
                    img = Image(img_path, width=max_width, height=max_height)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                else:
                    elements.append(Paragraph(f"[Image not found: {img_path}]", text_style))
            except Exception as e:
                elements.append(Paragraph(f"[Error loading image: {str(e)}]", text_style))
            
            # Add text after image
            if i+1 < len(parts) and parts[i+1]:
                elements.append(Paragraph(parts[i+1], text_style))
    
    # Handle case where there are more text parts than images
    for i in range(len(img_paths), len(parts)-1):
        if parts[i+1]:
            elements.append(Paragraph(parts[i+1], text_style))
    
    return elements

def process_prediction(text, img_paths, max_width=5*inch, max_height=4*inch):
    """Process prediction text and add images after the text"""
    elements = []
    
    # Add the text
    if text:
        elements.append(Paragraph(text, text_style))
    
    # Add a spacer
    elements.append(Spacer(1, 0.2*inch))
    
    # Add the images
    if img_paths:
        for img_path in img_paths:
            try:
                if os.path.exists(img_path):
                    img = Image(img_path, width=max_width, height=max_height)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                else:
                    elements.append(Paragraph(f"[Image not found: {img_path}]", text_style))
            except Exception as e:
                elements.append(Paragraph(f"[Error loading image: {str(e)}]", text_style))
    
    return elements

# Process each example
for i, example in enumerate(predictions_data):
    # Add page break between examples
    if i > 0:
        content.append(PageBreak())
    
    # Example header
    content.append(Paragraph(f"Example {i+1}", header_style))
    
    # Input section
    content.append(Paragraph("Input Text:", subheader_style))
    input_text = example.get("text", "No input text available")
    input_img_paths = example.get("input_img_paths", [])
    input_elements = process_text_and_images(input_text, input_img_paths)
    content.extend(input_elements)
    
    # Label section
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph("Label Text:", subheader_style))
    label_text = example.get("labels", "No label text available")
    label_img_paths = example.get("label_img_paths", [])
    label_elements = process_text_and_images(label_text, label_img_paths)
    content.extend(label_elements)
    
    # Prediction section
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph("Prediction Text:", subheader_style))
    pred_text = example.get("prediction", "No prediction text available")
    pred_img_paths = example.get("predicted_sketch_paths", [])
    
    # Process prediction differently - just add images after text
    pred_elements = process_prediction(pred_text, pred_img_paths)
    content.extend(pred_elements)

# Build the PDF
doc.build(content)
print(f"PDF saved to {output_pdf_path}")