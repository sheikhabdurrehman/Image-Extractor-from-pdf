import cv2
import numpy as np
from pdf2image import convert_from_path
import os

def calculate_energy(image):
    """Calculate the energy of an image based on the deviation from white."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the deviation from white (255 is white in grayscale)
    energy = 255 - gray

    return energy

def find_cropped_bounds(image, energy_threshold):
    """Find the bounding boxes of all regions of interest based on energy."""
    energy = calculate_energy(image)
    max_energy = np.max(energy)

    # Create a mask where energy is above the threshold
    mask = energy >= (energy_threshold * max_energy)
    
    # Find contours of the masked regions
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes for all contours
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    return bounding_boxes

def crop_images(image, energy_threshold=0.1, min_dimension=50):
    """Crop multiple images using the calculated bounds based on the energy threshold."""
    bounding_boxes = find_cropped_bounds(image, energy_threshold)
    cropped_images = []

    for (x, y, w, h) in bounding_boxes:
        # Check if the dimensions of the cropped image meet the minimum requirement
        if w >= min_dimension and h >= min_dimension:
            cropped_images.append(image[y:y + h, x:x + w])
    
    return cropped_images

def extract_images_from_pdf(pdf_path, output_folder, energy_threshold=0.1, min_dimension=50, poppler_path=None):
    """Extract images from each page of the PDF and crop regions of interest."""
    # Convert PDF to images
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    
    for i, page_image in enumerate(images):
        # Convert PIL image to OpenCV format
        page_image_cv = np.array(page_image)
        page_image_cv = cv2.cvtColor(page_image_cv, cv2.COLOR_RGB2BGR)

        # Crop regions of interest from the page image
        cropped_images = crop_images(page_image_cv, energy_threshold, min_dimension)

        # Save the cropped images
        for j, cropped_image in enumerate(cropped_images):
            output_image_path = os.path.join(output_folder, f'page_{i + 1}_cropped_{j + 1}.jpeg')
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Cropped image saved to {output_image_path}")

# Example usage
if __name__ == "__main__":
    pdf_path = r'your work 7 mini.pdf'  # Change to your PDF file path
    output_folder = r'chimagepdf7'  # Change to your output folder path
    poppler_path = r'C:\Users\92311\poppler-24.07.0\Library\bin'  # Change to your Poppler bin folder path

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract images from the PDF and crop them
    extract_images_from_pdf(pdf_path, output_folder, energy_threshold=0.1, min_dimension=100, poppler_path=poppler_path)
