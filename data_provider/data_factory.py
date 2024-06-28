from data_provider.data_loader import Dataset_Meteorology
from torch.utils.data import DataLoader, Subset

data_dict = {
    'Meteorology' : Dataset_Meteorology
}


def data_provider(args):
    Data = data_dict[args.data]

    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size 

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features
    )

    total_samples = len(data_set)
    val_size = int(args.split_ratio * total_samples)  
    train_size = total_samples - val_size  

    # 顺序划分数据集
    train_dataset = Subset(data_set, range(train_size))
    val_dataset = Subset(data_set, range(train_size, total_samples))
    print(f"val dataset length: {len(val_dataset)}")
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, train_data_loader, val_data_loader
