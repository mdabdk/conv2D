## 2D Convolution from Scratch

`conv2D.py` demonstrates 2D convolution on RGB images using basic NumPy operations, such as matrix multiplication, and compares the result to the result of the `scipy.signal.convolve2d()` function. Several kernels are available, including the Prewitt, Sobel, and Roberts kernels. Once `conv2D.py` is run, the user will be asked to enter a filename to process. For example, this could be `lena.png` or `../../path/to/other/image.png`.
