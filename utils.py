import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(logdir = log_dir)

    def log(self,Wasserstein_D, d_loss, g_loss, step):

        self.writer.add_scalar('{}/D_error'.format(self.log_dir), d_loss, step)
        self.writer.add_scalar('{}/G_error'.format(self.log_dir), g_loss, step)
        self.writer.add_scalar('{}/Wasserstein_D'.format(self.log_dir), Wasserstein_D, step)

    def log_images(self, images, num_images,step, format='NCHW'):
        '''
        input images are expected in format (NCHW)
        '''
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        
        if format=='NHWC':
            images = images.transpose(1,3)        

        img_name = '{}/images{}'.format(self.log_dir, '')

        horizontal_grid = vutils.make_grid(images)
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, step)

    def save_torch_images(self, horizontal_grid, step , plot_horizontal=True):
        
        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, step)
        plt.close()

    def _save_images(self, fig, epoch):
        if (epoch % 50) == 1 :
             out_dir = './output_folder/images'
        else :
             out_dir = './bin'

        Logger._make_dir(out_dir)
        fig.savefig('{}/iter_{}.png'.format(out_dir, epoch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        
        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
        
        
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch,num_epochs, n_batch, num_batches)
             )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './output_folder/models'
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),'{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),'{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality
    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise