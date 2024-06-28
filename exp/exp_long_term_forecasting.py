from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        data_set, train_data_loader, val_data_loader = data_provider(self.args)
        return data_set, train_data_loader, val_data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = f'runs/exp_long_term_forecast/{current_time}'
        writer = SummaryWriter(log_dir)
        total_data, train_loader, val_loader = self._get_data()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    # 打印当前iter loss 为什么不是平均loss？
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    global_step = epoch * len(train_loader) + i
                    # 对验证集验证mse mae
                    self.model.eval()
                    val_losses = []
                    val_mse = []
                    val_mae = []
                    with torch.no_grad(): 
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.float().to(self.device)
                            batch_y = batch_y.float().to(self.device)
                            if self.args.output_attention:
                                outputs = self.model(batch_x)[0]
                            else:
                                outputs = self.model(batch_x)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  
                            val_loss = criterion(outputs, batch_y)
                            mse = F.mse_loss(outputs, batch_y)
                            mae = F.l1_loss(outputs, batch_y)                  

                            val_losses.append(val_loss.item())
                            val_mse.append(mse.item())
                            val_mae.append(mae.item())
                    avg_val_loss = np.average(val_losses)
                    avg_mse = np.average(val_mse)
                    avg_mae = np.average(val_mae)
                    print(f"\tValidation Loss after iters {global_step+1}: {avg_val_loss:.7f}")
                    print(f"\tValidation MSE after iters {global_step+1}: {avg_mse:.7f}")
                    print(f"\tValidation MAE after iters {global_step+1}: {avg_mae:.7f}")
                    writer.add_scalar('\tLoss/validation', avg_val_loss, global_step)
                    writer.add_scalar('\tMSE/validation', avg_mse, global_step)
                    writer.add_scalar('\tMAE/validation', avg_mae, global_step)                        
                    writer.add_scalar('\tLoss/train', loss.item(), global_step)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch}.pth')
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        writer.close()
        return self.model