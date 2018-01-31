from shapeworld import dataset

dataset = dataset(dtype='agreement', name='oneshape_simple_textselect', config='load(/scratch/lhg256/comms/train)')
generated = dataset.generate(n=100, mode='train')

k = ['caption_str', 'texts_str', 'pred_items']

for i in range(10):
    print(f'Prediction items: {generated[k[2]][i]}, caption: {generated[k[0]][i]}, texts: {generated[k[1]][i]}')
