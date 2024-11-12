import os
import gradio as gr
from mistralai import Mistral
from PIL import Image
import io
from dotenv import load_dotenv
# Convert to base64 string
import base64

load_dotenv()

# Initialize Mistral client
def get_mistral_client():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    return Mistral(api_key=api_key)

def process_image(image):
    """Convert PIL Image to base64 string for Mistral"""
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    

    return base64.b64encode(img_byte_arr).decode('utf-8')

def transcribe_image(image, client):
    """Send image to Mistral and get LaTeX transcription"""
    base64_image = process_image(image)
    
    # Construct the prompt for the Mistral agent
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please transcribe this handwritten mathematical notation into LaTeX. Only provide the LaTeX code, no explanations."
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}" 
                }
            ]
        }
    ]
    
    # Get response from Mistral
    response = client.agents.complete(
        agent_id="ag:d40a8e90:20241105:mvastral:610ba98d",
        messages=messages
    )
    
    return response.choices[0].message.content

def transcribe_and_edit(image):
    """Main function for Gradio interface"""
    try:
        client = get_mistral_client()
        # First transcription
        latex_content = transcribe_image(image, client)
        return latex_content, latex_content
    except Exception as e:
        return f"Error: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(title="TexStral: Turn handwriting into LaTeX ✨") as demo:
    gr.Markdown("""
    # Handwriting to LaTeX Converter
    Upload an image containing mathematical handwriting and get it parsed into LateX code. Uses Mistral AI's Pixtral model!
    """)
    
    with gr.Row():
        # Left column for image upload
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image", 
                type="pil",
                height=400
            )
            convert_btn = gr.Button("Convert to LaTeX", variant="primary")
            
        # Right column for LaTeX output
        with gr.Column():
            # Raw LaTeX output
            latex_output = gr.Code(
                label="Generated LaTeX",
                language="markdown",
                lines=10
            )
            # Editable LaTeX
            latex_editor = gr.Textbox(
                label="Edit LaTeX",
                lines=10,
                max_lines=20,
                show_copy_button=True
            )
    
    # Button to trigger conversion
    convert_btn.click(
        fn=transcribe_and_edit,
        inputs=[image_input],
        outputs=[latex_output, latex_editor]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()