import os
import numpy as np

from models.model import CycleGAN
from speech_tools import load_pickle, sample_train_data

np.random.seed(300)

dataset = 'vcc2018'
src_speaker = 'VCC2SF3'
trg_speaker = 'VCC2TM1'
model_name = 'cyclegan_vc'
os.makedirs(os.path.join('experiments', dataset, model_name, 'checkpoints'), exist_ok=True)

data_dir = os.path.join('datasets', dataset)
exp_dir = os.path.join('experiments', dataset)

train_A_dir = os.path.join(data_dir, 'vcc2018_training', src_speaker)
train_B_dir = os.path.join(data_dir, 'vcc2018_training', trg_speaker)
exp_A_dir = os.path.join(exp_dir, src_speaker)
exp_B_dir = os.path.join(exp_dir, trg_speaker)

# Data parameters
sampling_rate = 22050
num_mcep = 24
frame_period = 5.0
n_frames = 128

# Training parameters
num_iterations = 400000
mini_batch_size = 1
generator_learning_rate = 0.0002
generator_learning_rate_decay = generator_learning_rate / 200000
discriminator_learning_rate = 0.0001
discriminator_learning_rate_decay = discriminator_learning_rate / 200000
lambda_cycle = 10
lambda_identity = 5

print('Loading cached data...')
coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
    os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
    os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))

model = CycleGAN(num_features=num_mcep, log_dir=os.path.join('experiments', dataset, model_name, 'runs'))

iteration = 1
while iteration <= num_iterations:
    dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)
    n_samples = dataset_A.shape[0]

    for i in range(n_samples // mini_batch_size):
        if iteration > 10000:
            lambda_identity = 0
        if iteration > 200000:
            generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
            discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size

        generator_loss, discriminator_loss = model.train(input_A=dataset_A[start:end], input_B=dataset_B[start:end],
                                                         lambda_cycle=lambda_cycle, lambda_identity=lambda_identity,
                                                         generator_learning_rate=generator_learning_rate,
                                                         discriminator_learning_rate=discriminator_learning_rate)

        if iteration % 10 == 0:
            print('Iteration: {:07d}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(iteration,
                                                                                                   generator_loss,
                                                                                                   discriminator_loss))
        if iteration % 5000 == 0:
            print('Checkpointing...')
            model.save(directory=os.path.join('experiments', dataset, model_name, 'checkpoints'),
                       filename='{}_{}.ckpt'.format(model_name, iteration))
        iteration += 1
