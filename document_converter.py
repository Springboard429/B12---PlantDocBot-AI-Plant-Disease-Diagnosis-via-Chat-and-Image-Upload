import os

def convert_document(input_path, output_dir):
    filename = os.path.basename(input_path)
    name = filename.replace(".pdf", ".md")

    output_path = os.path.join(output_dir, name)

    # Dummy conversion (for now)
    with open(output_path, "w") as f:
        f.write("# Converted Markdown\n\nThis is a sample output.")

    return output_path