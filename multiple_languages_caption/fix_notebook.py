import json

# Load the notebook
with open('train-italian.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with the forward method and fix it
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def forward(self, images, captions):' in source and 'hidden = image_features.unsqueeze(0)' in source:
            # Replace the line with the fixed version
            new_source = []
            for line in cell['source']:
                if 'hidden = image_features.unsqueeze(0)' in line:
                    # Replace with the fixed line and update the comment
                    new_line = line.replace(
                        'hidden = image_features.unsqueeze(0)  # Shape: [1, batch_size, hidden_dim]',
                        'hidden = image_features.unsqueeze(0).repeat(2, 1, 1)  # Shape: [2, batch_size, hidden_dim]'
                    )
                    new_source.append(new_line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

# Save the modified notebook
with open('train-italian-fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed and saved as train-italian-fixed.ipynb") 