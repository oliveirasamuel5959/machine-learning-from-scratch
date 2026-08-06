import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
  # Setup: where to find images and labels
  def __int__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.img_dir = os.path.join(root_dir, 'jpg')

    # load annotations
    labels = []
    self.labels = labels[0]
    
    # load transformations
    self.transform = transform
    
    # NEW: Keep track of any errors
    self.error_log = []
    
  # How many total samples
  def __len__(self):
    return len(self.anns)
  
  # How to get image and label number 'idx'
  def __getitem__(self, idx):
    # Load image with error handling
    try:
      # Normal loading code
      img_name = f'image_{idx:05d}.jpg'
      img_path = os.path.join(self.img_dir, img_name)
      image = Image.open(img_path)
      
      # Check for corruption
      image.verify() # verify() closes the file
      image = Image.open(img_path) # Reopen for transform use
      
      # Skip tiny images
      if image.size[0] < 32 or image.size[1] < 32:
        raise ValueError(f"Image too small: {image.size}")
      
      # Convert grayscale to RGB
      if image.mode != 'RGB':
        image = image.convert('RGB')
        
      label = self.labels[idx]
      
      if self.transform:
        image = self.transform(image)
    
    except Exception as e:
      # Log the issue instead of crashing
      self.error_log.append({
        'index': idx,
        'error': str(e),
        'path': img_path if 'img_path' in locals() else 'unknown'
      })

      print(f"Warning: Skipping corrupted image {idx}: {e}")
      # Try the next image (wrap around if needed)
      next_idx = (idx + 1) % len(self)
      
    return self.__getitem__(next_idx)
  
  def get_error_summary(self):
    # Review what went wrong after traning
    if not self.error_log:
      print("No errors encountered - dataset is clean!")
    else:
      print(f"\nEncountered {len(self.error_log)} problematic images:")
      for error in self.error_log[:5]: # Show first 5
        print(f"  Index {error['index']}: {error['error']}")
      if len(self.error_log) > 5:
        print(f"  ... and {len(self.error_log) - 5} more")
        
        
class MonitoredDataset(BaseDataset):
  # __init__...
  def __getitem__(self, idx):
    import time
    start_time = time.time()
    
    # Track how often each image is accessed
    self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
    
    # Load image using parent class (with error handling)
    result = super().__getitem__(idx)
    
    # Track how long it took
    load_time = time.time() - start_time
    self.load_times.append(load_time)
    
    # Warn if slow
    if load_time > 1.0:
      print(f"Slow load: Image {idx} took {load_time:.2f}s")
      
    return result
          
  def main():
    dataset = MonitoredDataset('./data')
    print(f"Total samples: {len(dataset)}")
    # dataset.print_stats()
    
    # Try loading one image
    img, label = dataset[0]