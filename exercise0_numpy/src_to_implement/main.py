from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def main():
    # checker pattern
    checker = Checker(resolution=250, tile_size=25)
    checker.draw()
    checker.show()
    
    # circle pattern
    circle = Circle(resolution=1024, radius=200, position=(512, 256))
    circle.draw()
    circle.show()
    
    # spectrum pattern
    spectrum = Spectrum(resolution=255)
    spectrum.draw()
    spectrum.show()

    # image generator
    generator = ImageGenerator(
        file_path='./data/exercise_data/',
        label_path='./data/Labels.json',
        batch_size=16,  
        image_size=[32, 32, 3],
        rotation=True,
        mirroring=True,
        shuffle=True
    )
    generator.show()


if __name__ == "__main__":
    main()
