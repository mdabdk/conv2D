# import NumPy library

import numpy as np

# import OpenCV library

import cv2 as cv

# import Signal library

from scipy import signal as dsp

# import test image based on user input

img_name = input(
"""Enter the name of the desired input image, including
file type extensions. For example: wolves.png. Make sure that this
image file is in the same directory as this code file.

Enter the name of your image here: """)
img_in = cv.imread(img_name)

# dictionary containing all the possible kernels

kernels_dict = {
        
        'box': (1/9)*np.array([[1,1,1],
                               [1,1,1],
                               [1,1,1]]), # box

        'FOD_x_pos': np.array([[-1,1]]), # first order derivative positive x
        
        'FOD_x_neg': np.array([[1,-1]]), # first order derivative negative x
        
        'FOD_y_pos': np.array([[-1],
                               [1]]), # first order derivative positive y
        
        'FOD_y_neg': np.array([[1],
                               [-1]]), # first order derivative negative y

        'Prewitt_x': np.array([[-1,0,1],
                               [-1,0,1],
                               [-1,0,1]]), # Prewitt x

        'Prewitt_y': np.array([[1,1,1],
                               [0,0,0],
                               [-1,-1,-1]]), # Prewitt y

        'Sobel_x': np.array([[-1,0,1],
                             [-2,0,2],
                             [-1,0,1]]), # Sobel x

        'Sobel_y': np.array([[1,2,1],
                             [0,0,0],
                             [-1,-2,-1]]), # Sobel y

        'Roberts_x': np.array([[0,1],
                               [-1,0]]), # Roberts x

        'Roberts_y': np.array([[1,0],
                               [0,-1]]) # Roberts y
        }

# asks the user for a kernel name

kernel_name = input(
"""Enter the name after the --> symbol for the desired kernel:

Box kernel --> box
Positive x first order derivative kernel --> FOD_x_pos
Negative x first order derivative kernel --> FOD_x_neg
Positive y first order derivative kernel --> FOD_y_pos
Negative y first order derivative kernel --> FOD_y_neg
Prewitt x kernel --> Prewitt_x
Prewitt y kernel --> Prewitt_y
Sobel x kernel --> Sobel_x
Sobel y kernel --> Sobel_y
Roberts x kernel --> Roberts_x
Roberts y kernel --> Roberts_y

Enter the name of the kernel here: """)
    
# provide user feedback
    
print('\nLoading...')

# the chosen kernel to be used later on

chosen_kernel = kernels_dict[kernel_name]

#------------------------------------------------------------------------------
# This function pads the input 'img' with the desired type of padding based on
# the pad_type string. The size of this padding depends on the shape of
# kernel_rot. The possible padding types are 'clip', 'wrap', 'copy', and
# 'reflect'

def pad_img(img,kernel_rot,pad_type):
    
    # get number of rows and columns to pad to image

    pad_size = (np.ceil((kernel_rot.shape[0]-1)/2).astype(int), # num of rows
                np.ceil((kernel_rot.shape[1]-1)/2).astype(int)) # num of columns
    
    # if no padding necessary, return the image unchanged
    
    if pad_size == (0,0):
        
        return img
    
    # initialize padded image padded image based on the amount of padding
    
    # if only one row is to be padded
    
    if pad_size == (1,0):
        
        img_pad = np.zeros((img.shape[0]+pad_size[0],
                        img.shape[1]+(pad_size[1])*2),dtype=np.float32)
        
        img_pad[1:None,:] = img
    
    # if only one column is to be padded
    
    elif pad_size == (0,1):
        
        img_pad = np.zeros((img.shape[0]+(pad_size[0])*2,
                        img.shape[1]+pad_size[1]),dtype=np.float32)
        
        img_pad[:,1:None] = img
    
    # if more than one row and more than one column to be padded
    
    else:
        
        img_pad = np.zeros((img.shape[0]+(pad_size[0])*2,
                        img.shape[1]+(pad_size[1])*2),dtype=np.float32)
        
        img_pad[1:-1,1:-1] = img
    
    # if pad_type is 'clip', then return already zero padded image
    
    if pad_type == 'clip':
        
        return img_pad
    
    # if pad_type is 'wrap', then modify padded zeros to wrap around image
    
    if pad_type == 'wrap':
        
        if pad_size == (1,0):
            
            img_pad[0,:] = img[-1,:] # first row
                    
        elif pad_size == (0,1):
            
            img_pad[:,0] = img[:,-1] # first column
                    
        else:
            
            img_pad[0,1:-1] = img[-1,:] # first row
            
            img_pad[1:-1,0] = img[:,-1] # first column
            
            img_pad[-1,1:-1] = img[0,:] # last row
        
            img_pad[1:-1,-1] = img[:,0] # last column
        
            img_pad[0,0] = img[-1,-1] # top-left corner
            
            img_pad[0,-1] = img[-1,0] # top-right corner
            
            img_pad[-1,0] = img[0,-1] # bottom-left corner
            
            img_pad[-1,-1] = img[0,0] # bottom-right corner
    
    # if pad_type is 'copy', then modify padded zeros to copy edges of image
    
    if pad_type == 'copy':
        
        if pad_size == (1,0):
            
            img_pad[0,:] = img[0,:] # first row
                    
        elif pad_size == (0,1):
            
            img_pad[:,0] = img[:,0] # first column
                    
        else:
            
            img_pad[0,1:-1] = img[0,:] # first row
            
            img_pad[-1,1:-1] = img[-1,:] # last row
            
            img_pad[1:-1,0] = img[:,0] # first column
            
            img_pad[1:-1,-1] = img[:,-1] # last column
            
            img_pad[0,0] = img[0,0] # top-left corner
            
            img_pad[0,-1] = img[0,-1] # top-right corner
            
            img_pad[-1,0] = img[-1,0] # bottom-left corner
            
            img_pad[-1,-1] = img[-1,-1] # bottom-right corner
    
    # if pad_type is 'reflect', then modify padded zeros to reflect edges of
    # image
    
    if pad_type == 'reflect':
        
        if pad_size == (1,0):
            
            img_pad[0,:] = img[1,:] # first row
                    
        elif pad_size == (0,1):
            
            img_pad[:,0] = img[:,1] # first column
                        
        else:
        
            img_pad[0,1:-1] = img[1,:] # first row
            
            img_pad[-1,1:-1] = img[-2,:] # last row
            
            img_pad[1:-1,0] = img[:,1] # first column
            
            img_pad[1:-1,-1] = img[:,-2] # last column
            
            img_pad[0,0] = img[1,1] # top-left corner
            
            img_pad[0,-1] = img[1,-2] # top-right corner
            
            img_pad[-1,0] = img[-2,1] # bottom-left corner
            
            img_pad[-1,-1] = img[-2,-2] # bottom-right corner
    
    return img_pad

#------------------------------------------------------------------------------
# This function performs the 2D convolution of the input image img_in with
# the kernel. The input image is padded based on the pad_type string

def conv2(img_in,kernel,pad_type):
    
    # initialize the output image based on the size of the input image
    
    img_out = np.zeros((img_in.shape[0],img_in.shape[1]),dtype=np.float32)
    
    # rotate kernel 180 degrees

    kernel_rot = np.rot90(kernel,2)
    
    # pad input image
    
    img_padded = pad_img(img_in,kernel_rot,pad_type)
    
    # loop through pixels in padded input image and store 2D convolution
    # results in the output image
    
    for x in range(img_in.shape[0]):
        for y in range(img_in.shape[1]):
            
            # perform elementwise multiplication of window of image with kernel
            
            img_out[x,y] = np.multiply(
                           kernel_rot,
                           img_padded[x:x+kernel_rot.shape[0],
                                      y:y+kernel_rot.shape[1]]
                           ).sum()
            
    return img_out

#------------------------------------------------------------------------------

# name of image channels

img_channel_name = ['Blue','Green','Red']

# initalize RGB image to be displayed later

filtered_img = np.zeros_like(img_in,dtype=np.float32)

# perform 2D convolution of each color channel with the kernel and compare the
# results with the built-in function dsp.convolve2D

for i in range(3):
    
    # compute 2D convolution with individual color channel using hand-made
    # conv2 function. Note that the pad_type, e.g. 'clip', needs to be changed
    # concurrently with the 'boundary' keyword in dsp.convolve2 to
    # successfully compare the two functions later on
    
    img_out = conv2(img_in[:,:,i],chosen_kernel,'clip')
    
    # compute 2D convolution with individual color channel using built-in
    # dsp.convolve2d function. Note that the 'boundary' keyworkd needs to be
    # changed concurrently with the pad_type in conv2 to successfully compare
    # the two functions later on
    
    conv2_true = dsp.convolve2d(
                                img_in[:,:,i],
                                chosen_kernel,
                                'same',
                                boundary='fill')
    
    # store results of hand-made conv2 function
    
    filtered_img[:,:,i] = np.copy(img_out)
    
    # show the output of the hand-made conv2 function
    
    print('\n{} channel conv2 output\n\n{}'.format(
            img_channel_name[i],img_out))
    
    # show the output of the built-in dsp.convolve2d function
    
    print('\n{} channel true 2D convolution\n\n{}'.format(
            img_channel_name[i],conv2_true))
    
    # compare the outputs of both functions. Note that the np.allclose function
    # is used instead of the np.equal function because in the case that the
    # box filter is used, which uses floating-point arithmetic, arrays of
    # floating point numbers will need to be compared, and this only possible
    # with the np.allclose function
    
    if np.allclose(img_out,conv2_true):
        print('\n{} channel good!'.format(img_channel_name[i]))
    else:
        print('\n{} channel not good'.format(img_channel_name[i]))

# convert the filtered image into a format that is suitable for display

filtered_img = np.copy(np.clip(filtered_img,0,255).astype(np.uint8))

# display filtered image

cv.imshow('Filtered Image',filtered_img)