import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImageConvolution:
    def __init__(self, image_path):
        # Načtení obrázku a převod na numpy array
        self.original_image = np.array(Image.open(image_path).convert('L'))
        self.height, self.width = self.original_image.shape

        # Definice 5 konvolučních jader
        #   1) Identity (neutrální)
        #   2) Sharpen (zvýraznění hran)
        #   3) Edge detect (detekce hran)
        #   4) Blur (rozmazání)
        #   5) Emboss (reliéf)
        self.kernels = {
            'identity': np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ]),
            'sharpen': np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]),
            'edge_detect': np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]),
            'blur': np.array([
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9]
            ]),
            'emboss': np.array([
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2]
            ])
        }

    def manual_convolution(self, image, kernel):
        # Příprava výstupního obrázku
        h, w = image.shape
        # Zvmenšení výstupního obrázku o 1px z každé strany
        output = np.zeros((h - 2, w - 2))

        # Ruční konvoluce
        for i in range(h - 2):
            for j in range(w - 2):
                # Výřez obrazu
                image_patch = image[i:i + 3, j:j + 3]
                # Výpočet konvoluce
                output[i, j] = np.sum(image_patch * kernel)

        # Normalizace hodnot
        output = np.clip(output, 0, 255)
        return output.astype(np.uint8)

    def apply_convolutions(self):
        # Aplikace všech konvolučních jader
        convolved_images = {}
        for name, kernel in self.kernels.items():
            convolved_images[name] = self.manual_convolution(
                self.original_image, kernel
            )
        return convolved_images

    def visualize_results(self, convolved_images):
        # Vizualizace výsledků
        plt.figure(figsize=(15, 3))

        # Původní obrázek
        plt.subplot(1, 6, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title('Původní')
        plt.axis('off')

        # Konvoluční jádra
        for i, (name, image) in enumerate(convolved_images.items(), start=2):
            plt.subplot(1, 6, i)
            plt.imshow(image, cmap='gray')
            plt.title(name)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('convolution_results.png')
        plt.close()


def main():
    # Cesta k obrázku - nahraďte vlastní cestou
    image_path = 'input_image.png'

    # Vytvoření instance a zpracování
    conv_processor = ImageConvolution(image_path)
    convolved_images = conv_processor.apply_convolutions()
    conv_processor.visualize_results(convolved_images)

    print("Konvoluce dokončeny. Výsledky uloženy do 'convolution_results.png'")


if __name__ == "__main__":
    main()
