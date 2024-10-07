# Face_Emotion_Detection

### ABSTRACT:

Humans have been different from other species of animals in one aspect of
showing accurate emotions on what we feel. It has always been a fascinating case
for various psychologists and scientists to study the way the human mind perceives
things and how we react to certain stimuli. On the same basis this document here,
is to show that we can program a computer to understand certain emotions using
facial recognition by adding images to the knowledge base and helping it grow its
understanding of emotions by learning from the said images in the knowledge
base. This helps us to feed information to the AI (artificial intelligence) to learn
from the knowledge base and help us figure out what emotions are being shown by
the chosen main subject at a particular frame at any given time. Emotion
recognition is one of the many facial recognition technologies that have developed
and grown through the years. Currently, facial emotion recognition software is
used to allow a certain program to examine and process the expressions on a
human’s face. Using advanced image dispensation, this software functions like a
human brain that makes it capable of recognizing emotions too. With this AI we
can detect and study different facial expressions to use with additional information
presented to us. This is useful for a variety of purposes, including investigations
and interviews, and allows authorities to detect the emotions of a person with just
the use of technology.


### BACKGROUND:

Emotion recognition has been booming over the years due to its versatile nature on
how it can impact our lives in a much more positive way. Due to the research done
in this field there are various ways emotion recognition can help us positively.
Some uses of emotion recognition in the daily world are as follows:-
Security measures: Emotion recognition is already used by schools and other
institutions since it can help prevent violence and improves the overall security of a
place.

<b>Hr assistance:</b> There are companies that use AI with emotion recognition API
capabilities as HR assistants. The system is helpful in determining whether the
candidate is honest and truly interested in the position by evaluating intonations,
facial expressions, keywords, and creating a report for the human recruiters for
final assessment.

<b>Customer service: </b> There are systems launched nowadays that are installed in
customer service centers. Using cameras equipped with artificial intelligence, the
customer’s emotions can be compared before and after going inside the center to
determine how satisfied they are with the service they’ve received. And if there is a
low score, the system can advise the employees to improve the service quality.

<b>Differently abled children:</b> There is a project using a system in Google Glass
smart glasses that aims to help autistic children interpret the feelings of people
around them. When a child interacts with other people, clues about the other
person’s emotions are provided using graphics and sound.

<b>Audience engagement:</b> Companies are also using emotion recognition to
determine their business outcomes in terms of the audience’s emotional responses.
Apple also released a new feature in their iPhones where an emoji is designed to
mimic a person’s facial expressions, called Animoji.

<b>Video game testing:</b> Video games are tested to gain feedback from the user to
determine if the companies have succeeded in their goals. Using emotion
recognition during these testing phases, the emotions a user is experiencing in
real-time can be understood, and their feedback can be incorporated in making the
final product.

<b>Healthcare:</b> The healthcare industry sure is taking advantage of facial emotion
recognition nowadays. They use it to know if a patient needs medicine or for
physicians to know who to prioritize in seeing first.


### METHODOLOGY:

Each image is a two-dimensional set of pixels. When we import an image using the
cv2 library, which is a library used for computer vision, each image pixel is
stored as a numpy array of (1x3) dimensions. These 3 numbers represent the RGB
(i.e. Red, Green, and Blue) colors as per the depth. Basically, this array is a
numerical representation of the color of that particular pixel. We then use binary
thresholding to make pixel values either zero or 255 based on the threshold
value provided. The pixel values below the threshold value are set to zero, and the
pixel values above the threshold value are set to 255. Here, 0 refers to black and
255 refers to white in a grayscale image.

After turning the original picture into the black and white image, we later detect
faces in the image and crop the image so that we do not have anything other than
the facial expression.

In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of
artificial neural network (ANN), most commonly applied to analyze visual
imagery. CNNs are also known as Shift Invariant or Space Invariant Artificial
Neural Networks (SIANN), based on the shared-weight architecture of the
convolution kernels or filters that slide along input features and provide
translation-equivariant responses known as feature maps. Counter-intuitively, most
convolutional neural networks are not invariant to translation, due to the
downsampling operation they apply to the input.

They have applications in image and video recognition, recommender systems,
image classification, image segmentation, medical image analysis, natural
language processing, brain–computer interfaces, and financial time series.

CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons
usually mean fully connected networks, that is, each neuron in one layer is
connected to all neurons in the next layer. The "full connectivity" of these networks
make them prone to overfitting data. Typical ways of regularization, or preventing
overfitting, include: penalizing parameters during training (such as weight decay)
or trimming connectivity (skipped connections, dropout, etc.) CNNs take a
different approach towards regularization: they take advantage of the hierarchical
pattern in data and assemble patterns of increasing complexity using smaller and
simpler patterns embossed in their filters. Therefore, on a scale of connectivity and
complexity, CNNs are on the lower extreme.

<b>Convolutional Layer</b>

This layer is the first layer that is used to extract the various features from the input
images. In this layer, the mathematical operation of convolution is performed
between the input image and a filter of a particular size MxM. By sliding the filter
over the input image, the dot product is taken between the filter and the parts of the
input image with respect to the size of the filter (MxM).
The output is termed as the Feature map which gives us information about the
image such as the corners and edges. Later, this feature map is fed to other layers
to learn several other features of the input image.

<b>Pooling Layer</b>

The Convolutional Layer is followed by a Pooling Layer. The primary aim of this
layer is to decrease the size of the convolved feature map to reduce the
computational costs. This is performed by decreasing the connections between
layers and independently operating on each feature map. It basically summarizes the
features generated by a convolution layer.

<b>Fully Connected Layer</b>

The Fully Connected (FC) layer consists of the weights and biases along with the
neurons and is used to connect the neurons between two different layers. These
layers are usually placed before the output layer and form the last few layers of a
CNN Architecture.

In this, the input image from the previous layers are flattened and fed to the FC
layer. The flattened vector then undergoes few more FC layers where the
mathematical functions operations usually take place. In this stage, the
classification process begins to take place. The reason two layers are connected is
that two fully connected layers will perform better than a single connected layer.
These layers in CNN reduce the human supervision

<b>Dropout</b>

Usually, when all the features are connected to the FC layer, it can cause overfitting
in the training dataset. Overfitting occurs when a particular model works so well on
the training data causing a negative impact in the model’s performance when used
on new data.

To overcome this problem, a dropout layer is utilized wherein a few neurons are
dropped from the neural network during the training process resulting in reduced
size of the model.

<b>Activation Functions</b>

Finally, one of the most important parameters of the CNN model is the activation
function. They are used to learn and approximate any kind of continuous and
complex relationship between variables of the network. In simple words, it decides
which information of the model should fire in the forward direction and which
ones should not at the end of the network.

It adds non-linearity to the network. There are several commonly used activation
functions such as the ReLU, Softmax, tanH and the Sigmoid functions. Each of
these functions have a specific usage. Here, we have used ReLU and Softmax for
our classification model along with categorical cross-entropy as the loss function.

#### RESULTS:
For this model, we have used ReLU and Softmax activation function for our
classification model along with categorical cross-entropy as the loss function and
we obtain the accuracy of 61.875% or 62%. The following graph plots the model
learning to predict emotions.

We later pass some images to the model for detecting and these are the results of
the subjects in the pictures.
