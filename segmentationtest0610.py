#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First_time: first time using this environment, need to install package
# ini_model: first time running the model
# Test: Output testing masks
# check: check preprocessing image
FIRST_TIME = False
INI_MODEL = True
TEST = True
CHECK = False



# In[6]:


# import packages
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


x_train_dir = 'data/images/train/rgb_images'
y_train_dir = 'data/images/train/gtLabels'

x_valid_dir = 'data/images/val/rgb_images'
y_valid_dir = 'data/images/val/gtLabels'

x_test_dir = 'data/image_test'


# In[8]:


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

            0 - void
            1 - road
            2 - lanemarks
            3 - curb
            4 - pedestrians
            5 - rider
            6 - vehicles
            7 - bicycle
            8 - motorcycle
            9 - traffic_sign
    
    """
    
    CLASSES = ['void', 'road', 'lanemarks', 'curb', 'pedestrians', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic_sign']

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            mode = 'train'
    ):
        self.ids = os.listdir(images_dir) # 返回指定的文件夹包含的文件或文件夹的名字的列表
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids] # list of image file path
        if mode == 'train':
          self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids] # list of annotation file path
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':  
            mask = cv2.imread(self.masks_fps[i], 0)

            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            '''
            # add background if mask is not binary
            if mask.shape[-1] != 1:
              background = 1 - mask.sum(axis=-1, keepdims=True)
              mask = np.concatenate((mask, background), axis=-1)
            '''
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image, mask
        if self.mode == "test": 
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']
                
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image= sample['image']
            return image

    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        # return tuple(batch)
        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# In[9]:


if CHECK:
    # Check the data and ground true mask
    dataset = Dataset(x_train_dir, y_train_dir, classes=['void', 'road', 'lanemarks', 'curb', 'pedestrians', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic_sign'])

    image, mask = dataset[5] # get some sample
    index = tf.argmax(mask, axis=2) # decode
    index = index.eval(session=tf.compat.v1.Session()) # to solve the error in tensor not np array

    print("image:",dataset.ids[5])
    visualize(
        image=image, 
        GTmask = index,
        people_mask=mask[..., 4].squeeze(),
        vehicles_mask=mask[..., 6].squeeze(),
        background_mask=mask[..., 0].squeeze(),
    )


# ### **Data augmentation**
# 

# In[10]:


import albumentations as A


# In[11]:


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [
        
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.7),
        A.GaussNoise(p=0.2),
        A.RGBShift(p=.9),
        A.RandomCrop(height=544, width=544, always_apply=True),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(1280, 1280, border_mode=cv2.BORDER_CONSTANT, value=0)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


# In[12]:


if CHECK:
    # Lets look at augmented data we have
    dataset = Dataset(x_train_dir, y_train_dir, classes=['void', 'road', 'lanemarks', 'curb', 'pedestrians', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic_sign'], augmentation=get_training_augmentation())

    image, mask = dataset[5] # get some sample
    index = tf.argmax(mask, axis=2) # decode
    index = index.eval(session=tf.compat.v1.Session()) # to solve the error in tensor not np array
    print("image:",dataset.ids[5])
    visualize(
        image=image, 
        GTmask = index,
        people_mask=mask[..., 4].squeeze(),
        vehicles_mask=mask[..., 6].squeeze(),
        background_mask=mask[..., 0].squeeze(),
    )


# ### **Modeling**

# In[13]:


import segmentation_models as sm
sm.set_framework('keras')
BACKBONE = 'efficientnetb5'
BATCH_SIZE = 2
CLASSES = ['void', 'road', 'lanemarks', 'curb', 'pedestrians', 'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic_sign']
# LR = 0.0001
LR = 1e-5
wd = 1e-6
EPOCHS = 15

preprocess_input = sm.get_preprocessing(BACKBONE)


# In[14]:


# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES))  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights = './weights/noisystudentb5.h5') # , encoder_weights = PRE_TRAINED_WEIGHTS

if not INI_MODEL:
    model.load_weights('weights/best_model_b5.h5')


# ### **Optimizer**

# In[15]:



# define optomizer
optim = keras.optimizers.Adam(LR, decay=wd)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss 

dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1., 1., 1., 1.,1.,1.,1.,1.,1.,])) 
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 
# total_loss = sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)


# Dataset

# In[16]:


if TEST is False:
    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        classes=CLASSES, 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 544, 544, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 544, 544, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./weights/best_model_b5.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]


# Training

# In[17]:


if TEST is False:
    # train model
    history = model.fit(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=valid_dataloader, 
        validation_steps=len(valid_dataloader),
    )


# In[18]:


if TEST is False:
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[19]:


if TEST:
# Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.load_weights('weights/best_model_b5.h5')
scores = model.evaluate_generator(valid_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))


# In[21]:

# crop back the image
aug = A.CenterCrop(p=1, height=966, width=1280)
if TEST:
    model.load_weights('weights/best_model_b5.h5')
    # Dataset for validation images
    '''
    test_dataset = Dataset(
        x_test_dir, 
        None, 
        classes=CLASSES, 
        preprocessing=get_preprocessing(preprocess_input),
        augmentation=get_validation_augmentation(),
        mode ='test'
    )
    '''

    test_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES, 
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    
    '''
    # n = 3
    # ids = np.random.choice(np.arange(len(test_dataset)), size=n)
    '''

    for i in range(0, len(test_dataset)):
    # for i in ids:
        print("image:",test_dataset.ids[i])
        # preprocessing
        image = test_dataset[i]
        image = np.expand_dims(image, axis=0) # shape = [1,1280,1280, 3]

        # predict the mask
        pr_mask = model.predict(image) # shape = [1280,1280, 10]
        pr_mask = pr_mask.squeeze() # shape = [1280,1280]

        # get the mask label
        index = tf.argmax(pr_mask, axis=2) # decode, # shape = [1280,1280]
        index = index.eval(session=tf.compat.v1.Session()) # to solve the error in tensor not np array

        # reduce dimension
        image = image.squeeze()  # shape = [1280,1280, 3]
        
        augmented = aug(image=image, mask=index)
        image = augmented['image']
        mask_predict = augmented['mask']  
        '''   
        visualize(
            image=denormalize(image),
            pr_mask=mask_predict,
        )
        '''
        
        mask_predict = np.dstack([mask_predict,mask_predict,mask_predict]).astype(np.uint8)
        cv2.imwrite("output0611/{}".format(test_dataset.ids[i]), mask_predict)
        del image, pr_mask, index, mask_predict





