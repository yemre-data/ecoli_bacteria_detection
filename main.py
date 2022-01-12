from dataset import EcoliBacteriaDataset
import torch.utils.data
data_folder = "/home/criuser/Desktop/Internship/Output"
train_dataset = EcoliBacteriaDataset(data_folder,
                                     train_test='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                           collate_fn=train_dataset.collate_fn, num_workers=4,
                                           pin_memory=True)

