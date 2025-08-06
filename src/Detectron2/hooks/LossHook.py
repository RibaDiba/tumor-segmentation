import matplotlib.pyplot as plt
import torch, os
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetMapper, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


"""
this file contains a custom hook that allows us to save and visualize loss data 
after each step we record the current loss data and after training we save everything

this is not a general use class and is specific to our use case 
(i.e specifically looking at loss_mask and total_loss)

every 100 iterations, we are going to look at the data and append it to the loss plot 

might make it more general later 
"""

class TrainingLossHook(HookBase):

    def __init__(self, output_dir, model_name, test_loader, cfg, save_data=True):
        self.save_data = save_data
        self.output_dir = output_dir
        self.eval_period = 50
        self.loss_dict_total_train = {}
        self.loss_dict_mask_train = {}
        self.cfg = cfg

        self.test_loader = test_loader

        self.model_name = model_name

        # make dir if doesnt exsist 
        os.makedirs(output_dir, exist_ok=True)
    
    """
    here we will get the loss at each step 
    bassically we are creating a dict whose key is the iteration and value is the loss
    this is so that we can create a line graph after
    """
    def after_step(self):
        storage = get_event_storage()
        iteration = self.trainer.iter

        current_scalars = storage.latest_with_smoothing_hint()
        total_loss = current_scalars["total_loss"][0]
        mask_loss = current_scalars["loss_mask"][0]

        self.loss_dict_total_train[iteration] = total_loss
        self.loss_dict_mask_train[iteration] = mask_loss

    """
    now we can create the plots and save them to the output dir 
    addtionally we can create a csv or json file with our dict 
    """
    def after_train(self):
        # for debug 
        print("Training Completed, now saving plots....")
        self._save_plots()
        if self.save_data: 
            self._save_data()
        
    def _save_plots(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Losses for {self.model_name} — {self.trainer.max_iter} iterations",
            fontsize=14
        )

        axs[0].plot(
            list(self.loss_dict_total_train.keys()),
            list(self.loss_dict_total_train.values()),
            label="train"
        )
        axs[0].set_title("Total Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(
            list(self.loss_dict_mask_train.keys()),
            list(self.loss_dict_mask_train.values()),
            label="train"
        )
        axs[1].set_title("Mask Loss")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        axs[1].grid(True)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(self.output_dir, f"{self.model_name}_loss_plot.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved loss curves to {out_path}")



    
    # TODO: finish this
    def _save_data(self):
        pass