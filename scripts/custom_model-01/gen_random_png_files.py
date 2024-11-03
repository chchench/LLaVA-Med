import os
import sys
from PIL import Image, ImageDraw
import random

def generate_abstract_image(width, height, shapes_per_image):
    """
    Generates an abstract geometric pattern image.

    Parameters:
    - width (int): Width of the image.
    - height (int): Height of the image.
    - shapes_per_image (int): Number of shapes to draw on the image.

    Returns:
    - Image object with the generated pattern.
    """
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    for _ in range(shapes_per_image):
        # Random shape type
        shape_type = random.choice(['rectangle', 'ellipse', 'line', 'polygon'])

        if shape_type in ['rectangle', 'ellipse']:
            # Generate coordinates ensuring x1 <= x2 and y1 <= y2
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            coords = [x1, y1, x2, y2]
        elif shape_type == 'line':
            # For lines, order doesn't matter, but we'll generate two points
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            coords = [x1, y1, x2, y2]
        elif shape_type == 'polygon':
            # Create a random polygon with 3 to 6 points
            num_points = random.randint(3, 6)
            coords = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_points)]
        else:
            # Default to a single point if shape_type is unrecognized
            coords = [random.randint(0, width), random.randint(0, height)]

        # Random color
        color = tuple(random.randint(0, 255) for _ in range(3))

        # Draw the shape
        if shape_type == 'rectangle':
            draw.rectangle(coords, fill=color, outline=None)
        elif shape_type == 'ellipse':
            draw.ellipse(coords, fill=color, outline=None)
        elif shape_type == 'line':
            line_width = random.randint(1, 5)
            draw.line(coords, fill=color, width=line_width)
        elif shape_type == 'polygon':
            draw.polygon(coords, fill=color, outline=None)

    return image

def main():
    # Check for the correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_images.py <output_directory> <max_number_of_files>")
        sys.exit(1)

    # Parse command-line arguments
    output_dir = sys.argv[1]
    max_files_str = sys.argv[2]

    # Validate and convert max_files to integer
    try:
        max_files = int(max_files_str)
        if max_files < 1:
            raise ValueError
    except ValueError:
        print("Error: <max_number_of_files> must be a positive integer.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Image dimensions and shapes per image
    width, height = 800, 800
    shapes_per_image = 50

    # Determine the number of digits for zero-padding based on max_files, minimum 2 digits
    num_digits = max(2, len(str(max_files)))

    for i in range(1, max_files + 1):
        # Generate the abstract image
        image = generate_abstract_image(width, height, shapes_per_image)

        # Generate the filename with leading zeros (e.g., image_001.png)
        filename = f"image_{i:0{num_digits}d}.png"
        filepath = os.path.join(output_dir, filename)

        # Save the image
        try:
            image.save(filepath)
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")

    print(f"All images have been generated and saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    main()
