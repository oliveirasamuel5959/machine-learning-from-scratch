import torchvision
import torchvision.transforms as transforms

def image_preprocessing(type='train'):
  
  if type == 'train':
    transform = transforms.Compose([
      # Random Agumentations (different each time)
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=10),
      transforms.ColorJitter(brightness=0.2),
      # transforms.Resize(256),
      # transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
  ])
    
  return transform