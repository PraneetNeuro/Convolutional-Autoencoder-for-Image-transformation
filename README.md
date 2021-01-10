# Convolutional-Autoencoder-for-Image-transformation
Can be used to build autoencoder models for image transformation like image colorization

# Note:
Feel free to tinker with the source file to work with images of different dimensions

## Sample code for training the model and inference
```python
dataset = Dataset('SRC_IMAGES_PATH', 'TARGET_IMAGES_PATH')
autoencoder = AutoEncoder(dataset, epochs=35)


def inference(images_path, save_path, ground_truth_path=None):
    if ground_truth_path is not None:
        for img_n in tqdm(os.listdir(images_path)):
            try:
                img = cv2.imread(images_path + img_n)
                img_ = np.array(cv2.resize(img, (100, 100))) / 255
                img = np.expand_dims(img_, 0)
                output = np.array(autoencoder.model.predict([np.array(img)])[0])
                target = cv2.imread(ground_truth_path + img_n)
                target = np.array(cv2.resize(target, (100, 100)))
                res = np.concatenate((img_ * 255, output * 255, target), axis=1)
                cv2.imwrite('{}/generated_{}.jpg'.format(save_path, os.path.splitext(img_n)[0]), res)
            except:
                pass
    else:
        output = np.array(autoencoder.model.predict([np.array(img)])[0])
        cv2.imwrite('{}/generated_{}.jpg'.format(save_path, os.path.splitext(img_n)[0]), output)
```
