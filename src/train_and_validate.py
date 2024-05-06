import torch
import anndata
from sklearn.model_selection import KFold
# from datasets.load_d4ls import load_full_anndata
from datasets.data_utils import load_full_anndata
from models.ModelBase import ModelBase
from models.xgboost import XGBoostModel
import argparse
from pathlib import Path
from models.vanilla_stellar import VanillaStellarReduced
from models.custom_stellar import CustomStellarReduced
import yaml

from models.torch_mlp import TorchMLP
from models.sklearn_mlp import SklearnMLP

CONFIG_PATH: Path = Path(__file__).parent / "config"

def main():
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument("--dataset-path", default="data/train", help="dataset path")
    parser.add_argument(
        "--method",
        default="stellar",
        choices=["stellar", "torch_mlp", "sklearn_mlp", "xgboost"]
    )
    parser.add_argument("--config", default="standard", help="Name of a configuration in src/config/{method} directory.")
    parser.add_argument("--cv-seed", default=42, help="Seed used to make k folds for cross validation.")
    parser.add_argument("--n-folds", default=5, help="Number of folds in cross validation.")
    parser.add_argument("--test", action="store_true", help="Test mode.")
    
    args = parser.parse_args()
    
    config = load_config(args)
    model = create_model(args, config)
    
    data = load_full_anndata(test=args.test)

    accuracy = cross_validation(data, model, random_state=args.cv_seed, n_folds=args.n_folds)
    
    print(accuracy)


def load_config(args):
    with open(CONFIG_PATH / args.method / f"{args.config}.yaml") as file:
        config = yaml.safe_load(file)

    return argparse.Namespace(**config)

def create_model(args, config) -> ModelBase:
    if args.method == "stellar":
        # return VanillaStellarReduced(config)
        return CustomStellarReduced(config)
    elif args.method == "torch_mlp":
        return TorchMLP(config)
    elif args.method == "sklearn_mlp":
        return SklearnMLP(config)
    elif args.method == "xgboost":
        return XGBoostModel(config)
    else:
        raise NotImplementedError(f"{args.method} method not implemented.")

def cross_validation(data: anndata.AnnData, model: ModelBase, random_state: int=42, n_folds: int=5):
    torch.manual_seed(42)

    accuracy_sum = 0.
    for i, (train_data, test_data) in enumerate(k_folds(data, n_folds, random_state)):
        model.train(train_data)
        pred = model.predict(test_data)
        
        accuracy = (pred == test_data.obs["cell_labels"]).sum() / len(pred)
        print(f"Validation accuracy of {i} fold:", accuracy)
        
        accuracy_sum += accuracy

    return accuracy_sum / n_folds

def k_folds(data: anndata.AnnData, n_folds: int, random_state: int):
    sample_ids = data.obs["sample_id"].cat.remove_unused_categories()
    sample_ids_unique = sample_ids.cat.categories

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    split = kfold.split(sample_ids_unique.tolist())
    
    for train, test in split:
        train_mask = data.obs["sample_id"].isin(sample_ids_unique[train])
        test_mask = data.obs["sample_id"].isin(sample_ids_unique[test])

        yield data[train_mask], data[test_mask]
        

if __name__ == "__main__":
    main()
