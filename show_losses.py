import pandas as pd
import matplotlib.pyplot as plt
from utils.Config import Config

df = pd.read_csv(f'{Config.RESULTS_DIR}/train/losses_logs.csv')

df.plot(y=[
    'G_loss', 'D_loss', 'L1', 'FeatureMatching', 'Perceptual', 
    'LPIPS', 'TotalVariation', 'GAN'
], figsize=(15, 10))
plt.title('Losses during training')
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.legend(loc='upper right')
plt.grid()
plt.show()

