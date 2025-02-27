import numpy as np
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice


class LDASplitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.5):
        self.alpha = alpha
        super(LDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset
        
        # Check if it's a HuggingFace dataset
        is_hf_dataset = hasattr(dataset, 'features') and hasattr(dataset, 'format')
        
        if is_hf_dataset:
            # For HuggingFace datasets, convert to pandas first to safely extract labels
            df = dataset.to_pandas()
            if 'label' in df.columns:
                label = df['label'].values
            elif 'categories' in df.columns:
                label = df['categories'].values
            else:
                raise ValueError('Cannot find label or categories in dataset')
                
            # Create index slices for data partitioning
            idx_slice = dirichlet_distribution_noniid_slice(label,
                                                            self.client_num,
                                                            self.alpha,
                                                            prior=prior)
            
            # Create subsets using HuggingFace's select method
            data_list = [dataset.select(idxs) for idxs in idx_slice]
            return data_list
        
        # Original code path for non-HuggingFace datasets
        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            if 'categories' in tmp_dataset[0]:
                label = np.array([x['categories'] for x in tmp_dataset])
            # added by me, for GLUE dataset
            elif 'label' in tmp_dataset[0]:
                label = np.array([x['label'] for x in tmp_dataset])
            else:
                raise ValueError('Cannot find label or categories in dataset')
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
