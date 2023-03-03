# total_loss = '1*L1+2*MSE+3*VGG+1*GAN'
# # total_loss = '1*L1+2*MSE+3*VGG+1*GAN'
# for loss in total_loss.split('+'):
#     weight, loss_type = loss.split('*')
#     if loss_type == 'MSE':
#         print('mse get')
#     elif loss_type == 'L1':
#         print('L1 get')
#     elif loss_type.find('VGG') >= 0:
#         print('vgg get')
#     elif loss_type.find('GAN') >= 0:
#         print('gan get')
a = '1-800/801-824'
train = 0
test = 1
data_range = [r.split('-') for r in a.split('/')]
if train:
    data_range = data_range[0]
else:
    if test and len(data_range) == 1:
        data_range = data_range[0]
    else:
        data_range = data_range[1]
print(data_range,len(data_range))