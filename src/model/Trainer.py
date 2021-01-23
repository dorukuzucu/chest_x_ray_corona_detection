import os
import torch
from torch import optim
from pathlib import Path


class Trainer:
    def __init__(self, model, criteria, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criteria = criteria
        self.config = config
        self.optimizer = self.set_optimizer()
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        self.RESULT_SAVE_PATH = os.path.join(Path(__file__).parents[2], "results") + os.path.sep
        self.gpu_flag = self.set_device()

    def train_supervised(self,n_epochs=None):

        if self.gpu_flag:
            self.model.cuda()
        if n_epochs is None:
            n_epochs = self.config["number_of_epochs"]
        self.model.train()

        for epoch in range(n_epochs):
            val_every = self.config["val_every"] if "val_every" in self.config else 5

            iter_loss = 0
            iter_correct_prediction = 0

            for data,label in self.train_loader:
                self.model.zero_grad()
                if self.gpu_flag:
                    data = data.cuda()
                    label = label.cuda()


                out = self.model(data)
                loss = self.criteria(out,label)

                loss.backward()
                self.optimizer.step()

                iter_loss+=loss
                iter_correct_prediction+=sum(out==label)
            self.metrics["train_loss"].append(iter_loss/len(self.train_loader))
            self.metrics["train_acc"].append(iter_correct_prediction/len(self.train_loader))
            print("Epoch:{} Loss:{} Accuracy:{}".format(epoch, self.metrics["train_loss"][epoch],
                                                        self.metrics["train_acc"][epoch]))

            if epoch % val_every == 0:
                val_loss = 0
                val_correct_prediction = 0
                self.model.eval()

                for data, label in self.val_loader:
                    self.model.zero_grad()
                    if self.gpu_flag:
                        data = data.cuda()
                        label = label.cuda()

                    out = self.model(data)
                    loss = self.criteria(out, label)

                    val_loss += loss
                    val_correct_prediction += sum(out == label)
                print("Epoch:{} Loss:{} Accuracy:{}".format(epoch, val_loss/len(self.val_loader),
                                                            val_correct_prediction/len(self.val_loader)))

            self.metrics["val_loss"].append(iter_loss / len(self.val_loader))
            self.metrics["val_accuracy"].append(iter_correct_prediction / len(self.val_loader))
            self.save_status()

    def set_optimizer(self):
        weight_decay = self.config["weight_decay"] if "weight_decay" in self.config else 0
        if self.config["optimizer"]=="SGD":
            momentum = self.config["momentum"] if "momentum" in self.config else 0
            self.optimizer = optim.SGD(
                params=self.model.parameters(),lr=self.config["lr"],
                weight_decay=weight_decay,momentum=momentum
            )
        else:
            self.optimizer = optim.Adam(
                params=self.model.parameters(),
                lr = self.config["lr"],
                weight_decay=weight_decay
            )

    def set_device(self):
        config_device = self.config["device"] if "device" in self.config else "cpu"
        if torch.cuda.is_available() and config_device=="gpu":
            self.gpu_flag=True
        else:
            self.gpu_flag=False

    def save_status(self):
        torch.save(self.model.state_dict(),self.RESULT_SAVE_PATH+"model.pth")
        self.save_results()

    def save_results(self):
        with open(self.RESULT_SAVE_PATH+"results.txt","w") as file:
            file.write(str(self.metrics)+"\n")
            file.close()