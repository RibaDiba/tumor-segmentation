import os, matplotlib.pyplot as plt
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage


"""
this file contains a custom hook that allows us to save and visualize loss data 
after each step we record the current loss data and after training we save everything

this is not a general use class and is specific to our use case 
(i.e specifically looking at loss_mask and total_loss)

might make it more general later 
"""

class LossVisualizationHook(HookBase):

    def __init__(self, output_dir, model_name, save_data=True):
        self.save_data = save_data
        self.output_dir = output_dir
        self.loss_dict_total = {}
        self.loss_dict_mask = {}
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

        self.loss_dict_total[iteration] = total_loss
        self.loss_dict_mask[iteration] = mask_loss

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
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"Losses for {self.model_name} - Trained with {self.trainer.max_iter} iterations")

        axs[0].plot(list(self.loss_dict_total.keys()), list(self.loss_dict_total.values()))
        axs[0].set_title("Total Loss Plot")
  
        axs[1].plot(list(self.loss_dict_mask.keys()), list(self.loss_dict_mask.values()))
        axs[1].set_title("Mask Loss Plot")

        fig.savefig(os.path.join(self.output_dir, f"{self.model_name}_plot.png"))

    # todo
    def _save_data(self):
        pass 
    

        