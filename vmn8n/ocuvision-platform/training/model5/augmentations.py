from torchvision import transforms

def train_transforms(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def val_transforms(size=224):
    return transforms.Compose([
        transforms.Resize(int(size*1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
