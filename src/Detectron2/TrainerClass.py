from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.data import transforms as T
from detectron2 import utils
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2, torch, os, json
import numpy as np

"""
this custom trainer class allows us to include image augmentations
also this is where we can create a hook to visualize training loss 
"""

class Trainer(DefaultTrainer):
    @classmethod 
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=tumor_mapper,
        )

augs = T.AugmentationList([
    T.RandomFlip(0.2, horizontal=True, vertical=False),
    T.RandomFlip(0.2, horizontal=False, vertical=True),
    T.RandomRotation([-30, 30], expand=False)
])

# custom mapper
def tumor_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    aug_input = T.StandardAugInput(image)
    transforms = augs(aug_input=aug_input)
    image = aug_input.image

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])

    return {
        "image": torch.as_tensor(image.transpose(2, 0, 1).astype("float32")),
        "instances": instances
    }

# custom hook for training loss
class LossVisualizationHook(HookBase):
    def __init__(self, loss_keys=None, output_dir='./loss_plots', save_data=True, output_json_dir="./json_Data"):
        self.save_data = save_data
        self.output_dir = output_dir
        self.output_json_dir = output_json_dir

        # default types of losses 
        self.loss_keys = [
            "total_loss", "loss_cls", "loss_box_reg",
            "loss_objectness", "loss_rpn_box"
        ]

        self.loss_history = defaultdict(list)
        self.iterations = []

        # create the output dir if it doesnt exis
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)

    """
    this function exectutes after each training step 
    we need to track our loss here by each step and graph after its done
    using total_losses here, could change 
    """
    def after_step(self):
        # get current iteration 
        iteration = self.trainer.iter
        storage = get_event_storage()

        # collect loss in a dict, we are storing all of them but only using one 
        current_losses = {}
        for key in self.loss_keys: 
            if key in storage._history and len(storage._history[key]) > 0:
                current_losses[key] = storage._history[key][-1][0]
        
        # only save if total loss is there
        if "total_loss" in current_losses: 
            self.iterations.append(iteration)
            for key, value in current_losses.items():
                self.loss_history[key].append(value)
    
    """
    this function runs after training is done 
    takes the data we recorded and plot/save it 
    """
    def after_train(self):
        # makes sure there is data 
        if len(self.iterations) > 1:
            print("Creating Loss Plots")
            # helper function
            self._plot_losses()
            if self.save_data:
                # helper function
                self._save_data()
        
        print(f"Plots have been saved to {self.output_dir}")
    
    def _plot_losses(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Losses Over Time', fontsize=16)
        
        axes = axes.flatten()
        plot_idx = 0
        
        for loss_name, loss_values in self.loss_history.items():
            if plot_idx < len(axes) and len(loss_values) > 0:
                ax = axes[plot_idx]
                ax.plot(self.iterations, loss_values, linewidth=2, label=loss_name)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.set_title(f'{loss_name.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                
                if len(loss_values) > 10:
                    window_size = min(50, len(loss_values) // 10)
                    smoothed = np.convolve(loss_values, 
                                         np.ones(window_size)/window_size, 
                                         mode='valid')
                    smoothed_iters = self.iterations[window_size-1:]
                    ax.plot(smoothed_iters, smoothed, '--', alpha=0.7, 
                           label=f'{loss_name} (smoothed)')
                
                ax.legend()
                plot_idx += 1
        
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'final_loss_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    # save data as json 
    def _save_loss_data(self):
        data = {
            'iterations': self.iterations,
            'losses': dict(self.loss_history)
        }

        with open(self.output_json_dir, 'w') as f:
            json.dump(data, f, indent=2)

        