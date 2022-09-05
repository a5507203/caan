import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv
from src.attack import FastGradientSignUntargeted
from src.utils import tensor2cuda, evaluate
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
class Tester():
    def __init__(self, model, dataset, args):
        """
            inputs
                model (nn.Module): a pretrained pytorch model
                dataset (torchdataste): pytorch dataset
                attack (string): types of attack
        """
        self.attack = None
        self.model = model
        if args.attack == "FastGradientSignUntargeted":
            self.attack = FastGradientSignUntargeted(model, 
                                                args.epsilon, 
                                                args.alpha, 
                                                min_val=0, 
                                                max_val=1, 
                                                max_iters=args.k, 
                                                _type=args.perturbation_type)

        self.loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def run(self):
        """
            return:
                accuracy (float): nature accuracy if attack=None, else robustness
        """

        # adv_test is False, return adv_acc as -1 
        self.model.eval()
        if self.attack == None:
            adv_test=False
            use_pseudo_label=True
        else: 
            adv_test = True
            use_pseudo_label = False

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in self.loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                self.model.eval()
                output = self.model(data)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', 
                                                       False)
                    self.model.eval()
                    adv_output = self.model(adv_data)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num



def main():
    ## an example
    model = resnet18(num_classes=10)
    model = model.cuda()
    dataset = CIFAR10(train=False,root="/.datasets",transform=tv.transforms.ToTensor(), download=True)
    import types
    args = types.SimpleNamespace()
    args.attack = "FastGradientSignUntargeted"
    args.epsilon = 0.0157
    args.alpha = 0.00784
    args.perturbation_type = "linf"
    args.batch_size = 128
    args.k = 10

    std_acc, adv_acc = Tester(model, dataset, args).run()
    print(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")

if __name__ == "__main__":
    main()








