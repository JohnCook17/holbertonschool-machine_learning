# 0x04. Data Augmentation

I have included a python notebook for easy viewing of the results.

    Task 0 flips an image using tf.image.flip_left_right.
===
    Task 1 crops an image using tf.image.random_crop, note the crop will be the
    same due to the seeds in the main file.
===
    Task 2 rotates the image 90 degrees counter clockwise using tf.image.rot90
===
    Task 3 was to shear the image, to make it work across multiple versions of
    Tensor Flow I had to make the process a little complicated. In short it uses
    the Keras ImageDataGenerator to shear the image, note due to the random seed
    not taking affect it may be inconsistent in its shearing to some degree.
===
    Task 4 adjust the brightness of an image using tf.image.adjust_brightness.
===
    Task 5 adjust the hue of an image using tf.image.adjust_hue
===
    Task 6 was to write a blog post here is the link
https://jcook0017.medium.com/automatic-data-augmentation-8bd0b4856a4f
===
    Task 100 was to preform PCA Color Augmentation, I based this off of
the work done by https://github.com/koshian2/PCAColorAugmentation and
changed pca_aug_numpy_single.py to work for my task. Principal Component
Analysis is a method of adjusting something so that only the important
information is shown. PCA Color Augmentation is a method to adjust the
color of an image using eigenvectors and eigenvalues. The easiest way to
think of it is like putting on sunglasses of different shades.
