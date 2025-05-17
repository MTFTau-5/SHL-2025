import yaml


def yaml_parser(config_path = '/home/mtftau-5/work3/SHL-2025/config/train.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        train_data_path = config['train_data_path']
        train_label_path = config['train_label_path']
        test_data_path = config['test_data_path']
        valid_data_path = config['valid_data_path']
        valid_label_path = config['valid_label_path']
        batch_size = config['batch_size']
        epochs = config['epochs']
        cnn_channels = config['cnn_channels']
        num_classes = config['num_classes']
        num_clusters = config['num_clusters']
        update_interval = config['update_interval']
        num_epochs = config['num_epochs']
        lr = config['lr']
        

   
    return ( 
        train_data_path, 
        test_data_path, 
        valid_data_path, 
        valid_label_path,
        train_label_path,
        batch_size,
        epochs,
        cnn_channels,
        num_classes,
        num_clusters,
        update_interval,
        num_epochs,
        lr
    )