
from preprocessing_GAN import get_clean_midi_data_as_images, clean_images, get_pixels_array
from model_GAN import def_discriminator, def_generator, def_gan, train

############################################################################
# preprocessing execution to get inputs for GAN model
############################################################################

# Selecting music pieces with one piano only and converting them to images
# Images stored in the 'data_image' folder
try:
    get_clean_midi_data_as_images('../../data_raw/', '../../data_image/')
    print("✅ images created")
except:
    print("❌ midi files transformation to images could not run")

# Cleaning the dataset of images
# Images stored in the 'data_image_cleaned' folder
try:
    clean_images('../../data_image/', '../../data_image_cleaned/')
    print("✅ images cleaned")
except:
    print("❌ images cleaning could not run")

# Creating an array of images to be used as dataset in Python for GAN model
# All images are stored one by one in an array of dimension n_images * 106 * 106 * 1
# 106 x 106 = image dimension / 1 = additional dimension needed for RNN input
try:
    pixels = get_pixels_array('../../data_image_cleaned/')
    print("✅ dataset as array available in 'pixels' variable")
except:
    print("❌ conversion of images into array could not run")


############################################################################
# GAN training execution
############################################################################

# Creating standalone discriminator model
try:
    discriminator_model = def_discriminator()
    print("✅ standalone discriminator created")
except:
    print("❌ standalone discriminator could not be created")

# Creating standalone generator model
try:
    generator_model = def_generator()
    print("✅ standalone generator created")
except:
    print("❌ standalone generator could not be created")

# Creating GAN model combining generator and discriminator
try:
    gan_model = def_gan(generator_model, discriminator_model)
    print("✅ GAN model created")
except:
    print("❌ GAN model could not be created")

# Training GAN model on pixels dataset
try:
    model, info = train(generator_model, discriminator_model, gan_model, pixels)
    print("✅ GAN model trained")
except:
    print("❌ training of GAN model could not run")
