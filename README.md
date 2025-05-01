# Training-Neural-networks-with-Keras



| Notebook Filename                                                   | Description                                                                                     |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `A_L1_and_L2_Regularization.ipynb`                                  | Compares CNNs with L1, L2, and no regularization to mitigate overfitting.                      |
| `B_Comparing_a_CNN_model_with_vs_without_Dropout.ipynb`             | Visualizes dropout effectiveness by training two CNNs with and without dropout layers.         |
| `C_Earlystopping.ipynb`                                             | Demonstrates the use of `EarlyStopping` to prevent overfitting by halting training early.      |
| `D_Montecarlo_Dropout.ipynb`                                        | Uses MC Dropout for uncertainty estimation by sampling predictions stochastically.             |
| `E_Comparing_different_Weight_Initializations_in_CNNs.ipynb`        | Compares `glorot_uniform`, `he_normal`, `lecun_normal`, and custom initializers in CNNs.       |
| `F_Batch_Norm.ipynb`                                                | Evaluates the impact of Batch Normalization on CNN training stability and generalization.      |
| `G_Custom_Dropout_and_Custom_Regularizer.ipynb`                     | Implements custom spatial dropout and threshold-based L1 regularization from scratch.          |
| `H_Callbacks_and_TensorBoard.ipynb`                                 | Shows how to use Keras callbacks including TensorBoard, ReduceLR, CSVLogger, and schedulers.   |
| `I_Keras_Tuner.ipynb`                                               | Uses Keras Tuner (Hyperband) to auto-tune model hyperparameters like layers, dropout, and LR.  |
| `J_KerasCV_Data_Augmentation_(CutMix_MixUp_+_A_B_Comparison).ipynb` | Applies KerasCV `CutMix` and `MixUp` and compares them against standard augmentation.          |
| `K_1_Image_Augmentation_and_CNN_Classification.ipynb`               | Applies traditional image augmentation on Fashion MNIST and trains a CNN classifier.           |
| `K_2_Video_Augmentation.ipynb`                                      | Demonstrates simple frame-wise augmentation techniques for video classification tasks.         |
| `K_3_Text_Augmentation.ipynb`                                       | Applies text augmentation techniques (e.g., synonym replacement, EDA) for NLP model training.  |
| `K_4_Timeseries_Augmentation_and_Classification.ipynb`              | Uses techniques like jittering, scaling, and permutation to augment time series data.          |
| `K_5_Tabular_Data_Augmentation_and_Classification.ipynb`            | Shows how to use SMOTE, Gaussian noise, and synthetic features to improve tabular models.      |
| `K_6_Speech_Augmentation_and_Classification.ipynb`                  | Augments audio using pitch shift, time stretch, and trains a speech classification model.      |
| `K_7_Document_Image_Augmentation_and_Classification.ipynb`          | Applies affine transforms, distortions, and noise to augment document image datasets.          |
| `L_FastAI_Augmentation.ipynb`                                       | Leverages FastAI to apply image augmentation and build a cat-vs-dog classifier on Oxford Pets. |


| Notebook Filename                                                                 | Description                                                                                          |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `01_Custom_LR_Scheduler.ipynb`                                                   | Implements custom learning rate schedulers including exponential decay and OneCycle policy.         |
| `02_Custom_Dropout.ipynb`                                                        | Creates a Monte Carlo-style `MCAlphaDropout` for stochastic inference and uncertainty estimation.    |
| `03_Custom_Normalization.ipynb`                                                  | Defines a `MaxNormDense` layer with max-norm kernel constraint to stabilize training.                |
| `04_Tensorboard.ipynb`                                                           | Demonstrates real-time training visualization with TensorBoard for model diagnostics.                |
| `05_Custom_Loss_Function.ipynb`                                                  | Implements Huber loss as a class and function, and explores custom threshold-based variants.         |
| `06_Custom_Activation_Function,Initializer_Regularizer_and_Kernel_Weight_Constraint.ipynb` | Combines `leaky_relu`, custom Glorot initializer, L1 regularizer, and positive weight constraint.    |
| `07_Custom_Metric_Huber.ipynb`                                                   | Creates `HuberMetric` as a custom streaming metric using `tf.keras.metrics.Metric`.                  |
| `08_Custom_Layers_Exponential_Layer_MyDense_GaussianNoise_LayerNormalization.ipynb` | Implements multiple custom layers: exponential activation, Gaussian noise, and layer normalization. |
| `09_Custom_Model_Residual_Block_and_Residual_Regressor.ipynb`                    | Builds a deep model using residual connections for stable regression learning.                      |
| `10_Custom_Optimizer.ipynb`                                                      | Implements a momentum-based optimizer from scratch, mimicking SGD with momentum.                    |
| `11_Custom_Training_Loop.ipynb`                                                  | Uses `tf.GradientTape` to build a custom training loop with metrics, progress bars, and LR control. |


[Training-Neural-networks-with-Keras](https://drive.google.com/file/d/1_cfXKdqKuWJqFrdhj0f4pkN-PKBkU-An/view?usp=drive_link)
