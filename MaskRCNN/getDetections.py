import torchvision.transforms as transforms
import numpy as np
import MaskRCNN.constants

# COCO category names for Mask R-CNN classificatons
COCO_CLASS_NAMES = MaskRCNN.constants.COCO_CLASS_NAMES

def getDetections(model, img, confidence, device):
  """
  getDetections
    parameters:
      - model: Mask R-CNN model
      - img: the input image as a numpy array
      - confidence: threshold to keep the prediction or not
      - device: cpu or cuda
    return:
      - masks, bounding boxes, and classes of detected object instances as numpy arrays
    method:
      - the image is converted to a tensor with a batch size of 1 and is passed 
        through the model to get the predictions
      - the predictions provide masks, bounding boxes, classes, and prediction scores
        - soft masks with continuous values are changed to binary masks where 1
          represents the object and 0 represents the rest of the image
        - bounding boxes are stored in a [[x0, y0], [x1, y1]] format
        - classes are stored as an array of COCO class names
        - prediction scores represent the classification probabilities
      - the prediction output is sorted by prediction scores, so the index of the
        last instance that meets the confidence criterion is used as a filter
      - bounding box coordinates are rounded to maximize area
  """
  model.to(device)

  transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Lambda(lambda x : x.unsqueeze(0))
      ])
  
  img = transform(img)
  img = img.to(device)
  pred = model(img)
  pred = pred[0]

  masks = (pred['masks'] > 0.5).squeeze().detach().cpu().numpy()
  boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().cpu().numpy())]
  classes = [COCO_CLASS_NAMES[i] for i in list(pred['labels'].detach().cpu().numpy())]

  pred_scores = list(pred['scores'].detach().cpu().numpy())
  last_inst_ind = [pred_scores.index(x) for x in pred_scores if x > confidence][-1]
  
  masks = masks[:last_inst_ind+1]
  boxes = np.array(boxes[:last_inst_ind+1])
  classes = np.array(classes[:last_inst_ind+1])

  boxes[:, 0] = np.floor(boxes[:, 0])
  boxes[:, 1] = np.ceil(boxes[:, 1])
  boxes = boxes.astype(int)

  return masks, boxes, classes