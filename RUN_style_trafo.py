#-----------------------------------------------------------------------------------------------------------------------------------
__author__ = "Christian Simonis"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Christian Simonis"
__email__ = "christian.Simonis.1989@gmail.com"
__status__ = "work in progress"
#-----------------------------------------------------------------------------------------------------------------------------------

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
#BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------------------------------------------------------------
# Name                          License  
# numpy                         BSD 3-Clause "New" or "Revised" License                   Copyright (c) 2005-2021, NumPy Developers: https://github.com/numpy/numpy/blob/main/LICENSE.txt
# tensorflow                    Apache Software License                                   Copyright 2019 The TensorFlow Authors.  All rights reserved, https://github.com/tensorflow/tensorflow/blob/master/LICENSE
# tensorflow-docs               Apache Software License                                   Copyright 2018 The TensorFlow Authors: https://github.com/tensorflow/docs/blob/master/LICENSE
# tensorflow-hub                Apache Software License                                   Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved,  https://github.com/tensorflow/hub/blob/master/LICENSE
# ipython                       BSD 3-Clause License                                      Copyright (c) 2008-Present, IPython Development Team, Copyright (c) 2001-2007, Fernando Perez <fernando.perez@colorado.edu>, Copyright (c) 2001, Janko Hauser <jhauser@zscout.de>, Copyright (c) 2001, Nathaniel Gray <n8gray@caltech.edu>, All rights reserved. https://github.com/ipython/ipython/blob/master/LICENSE                                                
# Pillow                        Historical Permission Notice and Disclaimer (HPND)        Copyright © 1997-2011 by Secret Labs AB, Copyright © 1995-2011 by Fredrik Lundh https://github.com/python-pillow/Pillow/blob/master/LICENSE
# matplotlib                    Python Software Foundation License,                       Copyright (c) 2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012 - 2021 The Matplotlib development team: https://matplotlib.org/stable/users/license.html
#-----------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Function, using TF API
runfile('apply_style_transfer.py') # call to define function "apply_style_transfer", used below. Wrapped code, based on tutorial from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/style_transfer.ipynb, Copyright 2018 The TensorFlow Authors.
#-----------------------------------------------------------------------------------------------------------------------------------
# Fundemental paremeter set

# Picture which is going to be used for content 
content_name = "Rome.jpg"

# List of styles which are learned and applied to content image via convolution layer
style_name = ["gondola.jpg", "sketch.jpg"]

# Input weights
style_weight=1e-2  # the higher the weight the more emphasis is on the style image
content_weight=1e4 # the higher the weight the more emphasis is on the content image
#-----------------------------------------------------------------------------------------------------------------------------------

# training specs 
epochs = 5 # for good results, pls. change to epochs >= 10


# Loop over style list
image_list = []
for idx, current_style in enumerate(style_name):

    
    # Map style to content picture
    print('____________________')
    print('Learning style from:')
    print(current_style)
    print('...              ...')
    image = apply_style_transfer(content_name,current_style ,style_weight, content_weight, epochs)
    image_list.append(image)
    

# visualize results
for idx in range(0,len(style_name)):

    plt.subplot(1, len(style_name), idx+1)
    plt.imshow(image_list[idx])
    plt.title('Style learned from: "' + style_name[idx] + '"', fontsize=12)
    
plt.show





