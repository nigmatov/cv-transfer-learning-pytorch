from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

def make_loaders(root: str, batch_size: int=32):
    tf_train = T.Compose([T.Resize((128,128)), T.RandomHorizontalFlip(), T.ToTensor()])
    tf_val = T.Compose([T.Resize((128,128)), T.ToTensor()])
    train_ds = ImageFolder(root=root, transform=tf_train)
    val_ds = ImageFolder(root=root, transform=tf_val)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size)
