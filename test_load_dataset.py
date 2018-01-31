from shapeworld import dataset

dataset = dataset(dtype='agreement', name='oneshape_simple_textselect', config='load(/scratch/lhg256/comms/test)')
generated = dataset.generate(n=100, mode='train')
