

### Dependencies

- python 3.8, pytorch, torch-geometric, torch-sparse, numpy, scikit-learn

If you have installed above mentioned packages you can skip this step. Otherwise run:

    pip install -r requirements.txt

## Switch data set

    python arguments.py 

## Setting parameters

Set parameters in file config/xx.yaml. 

## Reproduce data results

Load Data (run.py)

    model_save_dir = 'model_save'
    dataset_path = os.path.join(model_save_dir, args.DS, args.DS+'.pt')
    dataset = torch.load(dataset_path)

Load model (trainer.py)

    model_save_dir = 'model_save'
    contrastive_model_path = os.path.join(model_save_dir, self.name, 'contrastive_model_epoch.pth')
    tlc_model_path = os.path.join(model_save_dir, self.name, 'tlc_model_epoch.pth')
    self.contrastive_model.load_state_dict(torch.load(contrastive_model_path, map_location=self.device))
    self.tlc_model.load_state_dict(torch.load(tlc_model_path, map_location=self.device))

## Run

    python run.py 





