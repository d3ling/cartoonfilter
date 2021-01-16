import torch
import torchvision.transforms as transforms

def generateCartoon(model, input_img, dc_factor, device):
  """
  generateCartoon
    parameters:
      - model: CartoonGAN model
      - input_img: the input image as a numpy array
      - dc_factor: resizing factor due to down convolutions
      - device: cpu or cuda
    return:
      - cartoonized version of input_img as a numpy array
    method:
      - required image padding is determined to ensure image dimensions are
        compatible with down convolutions during CartoonGAN forward
      - input_img is transformed from numpy array to CartoonGAN input
        - RGB to BGR conversion
        - tensor conversion
        - normalize to range (-1, 1)
        - pad as necessary
        - create batch size of 1
      - output_img is transformed from CartoonGAN output to numpy array; this is
        an inverse of the previous step
  """
  model.to(device)
  
  h, w, _ = input_img.shape
  h_pad = (4 - (h % dc_factor)) % dc_factor
  w_pad = (4 - (w % dc_factor)) % dc_factor
  padding = (w_pad, h_pad, 0, 0) # pad on left, top, right and bottom borders 

  transform = transforms.Compose([
          transforms.Lambda(lambda x : x[:, :, [2, 1, 0]]),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          transforms.Pad(padding),
          transforms.Lambda(lambda x : x.unsqueeze(0))
      ])

  inverse = transforms.Compose([
          transforms.Lambda(lambda x : x[0]),
          transforms.Lambda(lambda x : x[:, h_pad:, w_pad:]),
          transforms.Lambda(lambda x : x * 0.5 + 0.5),
          transforms.Lambda(lambda x : (x * 255).int().permute(1, 2, 0).byte().detach().cpu().numpy()),
          transforms.Lambda(lambda x : x[:, :, [2, 1, 0]])
      ])

  input_img = transform(input_img)
  input_img = input_img.to(device)

  with torch.no_grad():
    output_img = model(input_img)

  output_img = inverse(output_img)

  return output_img