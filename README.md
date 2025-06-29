# Task B: Face Recognition with Triplet Loss

## Overview
This project implements a face recognition system using a ResNet-18 backbone trained with triplet loss. The task is part of Comys Hackathon 5 (Task B) and is designed to learn face embeddings that ensure same-person images are close in feature space and different-person images are distant.

## TripletFaceDataset

`TripletFaceDataset` is a custom PyTorch `Dataset` designed for training face-recognition models using triplet loss. It organizes images of each person into “anchor”, “positive”, and “negative” triplets:

1. **Initialization**  
   - Scans a root directory where each subfolder represents one person (identified by `person_id`).  
   - Collects all `.jpg` images in each person’s folder, and (optionally) any additional images in a `distortion` subfolder.  
   - Builds a list of valid persons (those with at least two images) and maps each person to their image paths.  

2. **Triplet Sampling**  
   - For each person, splits their images into:  
     - **Anchors**: original (undistorted) photos.  
     - **Positives**: images from the `distortion` subfolder whose filenames correspond to an anchor.  
   - Generates one triplet per (anchor, positive) pair by randomly sampling a **negative** image from a different person.  
   - Stores all `(anchor, positive, negative)` tuples in `self.samples`.  

3. **`__len__`**  
   Returns the total number of triplets available.

4. **`__getitem__`**  
   - Loads the anchor, positive, and negative images from disk and converts them to RGB.  
   - Assigns an `anchor_label` based on the anchor’s folder name (ignoring any `distortion` directory).  
   - Applies any provided `transform` (e.g., normalization, data augmentation) to each image.  
   - Returns a tuple:  
     ```python
     (anchor_image, positive_image, negative_image, anchor_label)
     ```  

This structure makes it easy to feed batches of triplets into a model and compute triplet loss, encouraging the network to pull matching faces closer in embedding space while pushing different faces apart.

### Image Transform

The dataset uses the following transform pipeline to preprocess each image before feeding it into the model:

1. **Rescales each image to a fixed 256×256 resolution.**
2. **Converts the PIL image to a PyTorch tensor and scales pixel values to the [0, 1] range.**


## Model
- **Backbone**: wraps a ResNet-18 backbone to produce fixed-size embeddings.
- **Embedding dimension**: Dimensionality of the output feature vector (default: 128).
- **Architecture adjustments**: Replaces ResNet-18’s final fully connected layer (fc) with a new nn.Linear that maps to embedding_size.
- **Loss**: Triplet margin loss.

## Training
Training is run for 12 epochs. For each epoch:
1. Compute triplet loss over the training batches.
2. Track metrics: Loss, Accuracy, Precision, Recall, F1-score.
3. Save model checkpoints.
4. Evaluate model on evaluation dataset.
5. Track evaluation metrics: Loss, Accuracy, Precision, Recall, F1-score.

## Results

| Epoch | Phase       | Loss     | Accuracy | Precision | Recall   | F1-score |
|-------|-------------|----------|----------|-----------|----------|----------|
| 1     | Training    | 0.046505 | 0.678757 | 0.608902  | 0.999481 | 0.756767 |
|       | Evaluation  |    —     | 0.761002 | 0.677079  | 0.997969 | 0.806787 |
| 2     | Training    | 0.001564 | 0.744140 | 0.661498  | 1.000000 | 0.796267 |
|       | Evaluation  |    —     | 0.769634 | 0.685103  | 0.997969 | 0.812457 |
| 3     | Training    | 0.000305 | 0.754525 | 0.670713  | 1.000000 | 0.802906 |
|       | Evaluation  |    —     | 0.777928 | 0.692904  | 0.998307 | 0.818031 |
| 4     | Training    | 0.000174 | 0.759828 | 0.675519  | 1.000000 | 0.806340 |
|       | Evaluation  |    —     | 0.774204 | 0.689252  | 0.998646 | 0.815593 |
| 5     | Training    | 0.000109 | 0.763574 | 0.678955  | 1.000000 | 0.808783 |
|       | Evaluation  |    —     | 0.775051 | 0.689881  | 0.999323 | 0.816259 |
| 6     | Training    | 0.000128 | 0.766986 | 0.682115  | 1.000000 | 0.811021 |
|       | Evaluation  |    —     | 0.781483 | 0.696155  | 0.998984 | 0.820520 |
| 7     | Training    | 0.000081 | 0.766911 | 0.682046  | 1.000000 | 0.810972 |
|       | Evaluation  |    —     | 0.780298 | 0.695007  | 0.998984 | 0.819722 |
| 8     | Training    | 0.000065 | 0.770546 | 0.685444  | 1.000000 | 0.813369 |
|       | Evaluation  |    —     | 0.781652 | 0.696319  | 0.998984 | 0.820634 |
| 9     | Training    | 0.000047 | 0.776072 | 0.690676  | 1.000000 | 0.817041 |
|       | Evaluation  |    —     | 0.784699 | 0.699384  | 0.998646 | 0.822644 |
| 10    | Training    | 0.000027 | 0.777481 | 0.692023  | 1.000000 | 0.817983 |
|       | Evaluation  |    —     | 0.782160 | 0.696905  | 0.998646 | 0.820927 |
| 11    | Training    | 0.000022 | 0.776591 | 0.691172  | 1.000000 | 0.817388 |
|       | Evaluation  |    —     | 0.786561 | 0.701117  | 0.998984 | 0.823956 |
| 12    | Training    | 0.000038 | 0.781969 | 0.696348  | 1.000000 | 0.820997 |
|       | Evaluation  |    —     | 0.796378 | 0.710913  | 0.998984 | 0.830683 |

## Evaluation
At each epoch, after training, the model is evaluated on the validation set using a fixed threshold (default 0.6). The evaluation script outputs

### Inputs:

**custom_model**: your embedding network (e.g., EmbeddingNet).

**loader**: DataLoader yielding (anchor, positive, negative, label) triplets.

**device**: computation device, either 'cpu' or 'cuda'.

**threshold**: cosine-similarity cutoff for deciding a “match” (default 0.5).

### Procedure:

**No gradients**: wraps evaluation in torch.no_grad().

**Batch loop**: for each triplet, move images to device and compute embeddings.

### Similarity scores:

sim_ap = cosine similarity between anchor & positive.

sim_an = cosine similarity between anchor & negative.

**Thresholding**: convert similarities ≥ threshold into positive predictions (1), else negative (0).

Accumulate true labels (1 for anchor-positive, 0 for anchor-negative) and predicted labels.

**Metrics**: compute accuracy, precision, recall, and F1-score.

**Progress logging**: prints percentage complete and final metrics.
