import base64
from io import BytesIO
from PIL import Image
import gradio as gr
from rameshm.llmeng.create_llm_instance import LLM_Instance

llm_instance = LLM_Instance("gpt-4o-mini")

def artist(model_nm,prompt):
    # OpenAI has two ways to create images: ResponsesAPI and ImagesAPI.
    # Here we are just creating an image, hence using ImagesAPI. ResponsesAPI is more relevant for back and forth on image
    # generation.
    print(f"Using Model: {model_nm}")
    if not prompt:
        city = "Paris"
        prompt = f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style"

    if "gpt-image" in model_nm:
        image_response = llm_instance.get_llm_model_instance().images.generate(
                model=model_nm, # gpt-image-1 or dall-e-3 or dall-e-2
                prompt=prompt,
                size="1024x1024", # there are other options to represent in landscape or even a default option of "auto"
                quality="medium", # options are high, medium, low, auto
                n=1, # One image
                #response_format="b64_json",
            )
    elif "dall-e" in model_nm:
        image_response = llm_instance.get_llm_model_instance().images.generate(
            model=model_nm,  # gpt-image-1 or dall-e-3 or dall-e-2
            prompt=prompt,
            size="1024x1024",  # there are other options to represent in landscape or even a default option of "auto"
            quality="standard", # options are "standard", "hd"
            n=1,  # One image
            response_format="b64_json",
        )
    else:
        raise ValueError(f"{model_nm} is not supported")

    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

display_image = gr.Interface(
    fn=artist,
    inputs=[
        gr.Dropdown(label="Model", choices=["dall-e-3", "dall-e-2"], value="dall-e-3"),
        gr.Textbox(label="Prompt", placeholder="Enter your prompt for image here...", interactive=True)
    ],
    outputs=[
        gr.Image(image_mode="RGB") # Deliberately skipped providing label so that label doesn't overlap image
    ],
    flagging_mode="never"
)

if __name__ == "__main__":
    display_image.launch(inbrowser=True)
    # mode = input("What do you want to generate: Enter either 'image' and 'sound': ")
    # mode = mode.lower()
    # if mode == "image":
    #     display_image.launch(inbrowser=True)
    # elif mode == "sound":
    #     pass
    # else:
    #     print(f"{mode} : Is not supported")
