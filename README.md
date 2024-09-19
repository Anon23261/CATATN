from PIL import Image

def encode_message(image_path, message, output_path):
    img = Image.open(image_path)
    encoded = img.copy()
    width, height = img.size
    index = 0

    for row in range(height):
        for col in range(width):
            if index < len(message):
                r, g, b = img.getpixel((col, row))
                ascii_value = ord(message[index])
                encoded.putpixel((col, row), (r, g, ascii_value))
                index += 1
            else:
                break

    encoded.save(output_path)
    print(f"Message encoded and saved to {output_path}")

def decode_message(image_path):
    img = Image.open(image_path)
    width, height = img.size
    message = ""

    for row in range(height):
        for col in range(width):
            r, g, b = img.getpixel((col, row))
            if b != 0:
                message += chr(b)
            else:
                break

    return message

# Example usage
image_path = "input_image.png"  # Path to the input image
output_path = "output_image.png"  # Path to save the encoded image
message = "I love you!"  # Secret message to encode

encode_message(image_path, message, output_path)
decoded_message = decode_message(output_path)
print(f"Decoded message: {decoded_message}")
