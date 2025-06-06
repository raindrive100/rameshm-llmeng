import gradio as gr

""" Example 1 : Simple Gradio interface to greet a user with a name and intensity"""
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity), f"Intensity: {intensity}"

demo_greet = gr.Interface(
    fn=greet,
    inputs = [
        gr.Textbox(label="Enter Name", lines=5, placeholder="Enter your name here ...", info="First Name please"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Intensity", value=5,),
    ],
    outputs = [gr.Textbox(label="Greeting", lines=3), "text"],
    title="Fun Greeter",
    description="Here's a sample Greeter.",
)

""" Example 2 to pass session history in Gradio """
def store_message(message: str, history: list[str]):
    output = {
        "Current messages": message,
        "Previous messages": history[::-1]  # Reverse the history for better readability
    }
    history.append(message)
    return output #, history

demo_chat = gr.Interface(fn=store_message,
                    inputs=["textbox", gr.State(value=[])],
                    outputs=["json", gr.State()])

""" Example-3 : make interfaces automatically refresh by setting live=True in the interface.
    Now the interface will recalculate as soon as the user input changes)
"""
def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2
    else:
        return "Invalid operation"

demo_calculator = gr.Interface(
    calculator,
    [
        "number",
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    "number",
    live=True,
)

""" Example function of Hello using Blocks instead of Interface """
def greet_block(name):
    return "Hello " + name + "!"

with gr.Blocks() as demo_greet_block:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet_block, inputs=name, outputs=output, api_name="greet")

""" Example 4 of function returning a list and dictionary using Gradio blocks """
with gr.Blocks() as demo_blocks_list:
    food_box = gr.Number(value=10, label="Food Count")
    status_box = gr.Textbox()

    def eat_list(food):
        if food > 0:
            return food - 1, "full"
        else:
            return 0, "hungry"

    def eat_dict(food):
        if food > 0:
            return {food_box: food - 1, status_box: "full"}
        else:
            return {food_box: food - 1, status_box: "hungry"} #  see how we can return just one element and not 2 like above

    gr.Button("Eat_return_list").click(
        fn=eat_list,
        inputs=food_box,
        outputs=[food_box, status_box]
    )
    gr.Button("Eat_return_dict").click(
        fn=eat_dict,
        inputs=food_box,
        outputs=[food_box, status_box]
    )

# Example 5. To Launch the Gradio Chat Interface
def llm_chat_example():
    gr.load_chat("http://localhost:11434/v1/", model="llama3.2", token="ollama").launch()
    gr.load_chat("http://localhost:11434/v1/", model="gemma3:1b", token="ollama").launch()

def main(function_name: str):
    if function_name == "greet":
        demo_greet.launch()
    elif function_name == "chat":
        demo_chat.launch()
    elif function_name == "calculator":
        demo_calculator.launch()
    elif function_name == "greet_block":
        demo_greet_block.launch()
    elif function_name == "blocks_list":
        demo_blocks_list.launch()
    elif function_name == "llm_chat":
        llm_chat_example()

if __name__ == "__main__":
    import sys
    main("greet")  # Change to "greet" or "chat" to run those examples)