import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load model
model_name = "Qwen/Qwen1.5-0.5B"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cuda", 
    torch_dtype=torch.float16
)

# Extract weights
layer_module = model.model.layers[0].mlp.gate_proj
weight_matrix = layer_module.weight.detach().cpu().numpy()

# Save weights
np.savetxt("weights_W.txt", weight_matrix, fmt="%.6f")
print("Saved weights.")

# Hook
activation_X = None
activation_Y = None

def hook_fn(module, input, output):
    global activation_X, activation_Y
    activation_X = input[0].detach().cpu().numpy()
    activation_Y = output.detach().cpu().numpy()

hook_handle = layer_module.register_forward_hook(hook_fn)

# Run inference
text_input = "FPGA design"
inputs = tokenizer(text_input, return_tensors="pt").to("cuda")

print("Calculating...")
with torch.no_grad():
    outputs = model(**inputs)

# Save X and Y
activation_X_2d = activation_X.reshape(-1, activation_X.shape[-1])
activation_Y_2d = activation_Y.reshape(-1, activation_Y.shape[-1])

np.savetxt("input_X.txt", activation_X_2d, fmt="%.6f")
np.savetxt("output_Y.txt", activation_Y_2d, fmt="%.6f")

print("Saved inputs and outputs.")
hook_handle.remove()
print("Done!")