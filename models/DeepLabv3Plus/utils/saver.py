import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, config):
        self.config = config
#        self.directory = os.path.join('run', self.config['dataset'], self.config['checkname'])
        self.directory =  self.config["training"]["output_dir"]

#        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
#        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
#        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))

        self.experiment_dir = self.config["training"]["output_dir"]

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

            """                
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            """

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.config['dataset']['dataset_name']
        p['backbone'] = self.config['network']['backbone']
        p['out_stride'] = self.config['image']['out_stride']
        p['lr'] = self.config['training']['lr']
        p['lr_scheduler'] = self.config['training']['lr_scheduler']
        p['loss_type'] = self.config['training']['loss_type']
        p['epoch'] = self.config['training']['epochs']
        p['base_size'] = self.config['image']['base_size']
        p['crop_size'] = self.config['image']['crop_size']

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()